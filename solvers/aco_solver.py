from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple, Set

# Import các hàm và cấu trúc dữ liệu chuẩn từ môi trường simulation
from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]  # cargo_op: 0 (không làm gì), 1 (nhặt hàng), 2 (giao hàng)

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class ACOSolver(Solver):
    """
    Ant Colony Optimization Policy hoàn chỉnh cho Online MAPD.
    - Sửa đổi chính xác Chi phí tải trọng phân tầng (trước/sau khi nhặt hàng).
    - Tuân thủ tuyệt đối luật tương tác Gym Env (chỉ phát op khi đứng yên tại ô đích).
    - Giải quyết tranh chấp ô trống theo độ ưu tiên ID tăng dần của đề bài.
    """

    method_name = "ACO_Final_Edition"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._path_cache: Dict[Tuple[Position, Position], Tuple[int, Move]] = {}
        self._pheromone: Dict[Tuple[str, int], float] = {}
        
        # Bộ siêu tham số ACO tối ưu hóa qua thực nghiệm
        self._alpha = 1.2      # Trọng số quyết định của Pheromone
        self._beta = 2.0       # Trọng số quyết định của Heuristic (Net Profit)
        self._rho = 0.05       # Tốc độ bốc hơi Pheromone
        self._deposit = 1.0    # Lượng Pheromone để lại khi tìm thấy đường tốt
        self._min_pheromone = 0.1
        self._max_pheromone = 10.0

        # Bản đồ mật độ đơn hàng động (Thích nghi Surge/Hotspot mù ở Phase 1)
        self._dynamic_hotspots: Dict[Position, float] = {}

    # ------------------------------------------------------------------
    # Grid & BFS Pathfinding Utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}

        while queue:
            current = queue.popleft()
            if current == goal:
                return parent

            for move, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = (current, move)
                queue.append(nxt)

        return None

    def _compute_path_properties(self, start: Position, goal: Position) -> Tuple[int, Move]:
        if start == goal:
            return 0, "S"

        key = (start, goal)
        if key in self._path_cache:
            return self._path_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._path_cache[key] = (INF, "S")
            return INF, "S"

        distance = 0
        current = goal
        first_move = "S"

        while current != start:
            previous, move = parent[current]
            if previous is None:
                self._path_cache[key] = (INF, "S")
                return INF, "S"
            if previous == start:
                first_move = move
            current = previous
            distance += 1

        self._path_cache[key] = (distance, first_move)
        return distance, first_move

    def _distance(self, start: Position, goal: Position) -> int:
        return self._compute_path_properties(start, goal)[0]

    def _next_move(self, start: Position, goal: Position) -> Move:
        return self._compute_path_properties(start, goal)[1]

    # ------------------------------------------------------------------
    # Tính toán chính xác Chi phí tải trọng di chuyển theo bước
    # ------------------------------------------------------------------
    def _estimated_travel_cost(self, distance: int, current_w: float, max_w: float) -> float:
        if distance >= INF: return INF * 0.01
        # Công thức đề bài: rc = -0.01 * (1 + W_carried / W_max) cho mỗi bước đi
        return distance * 0.01 * (1.0 + (current_w / max(1.0, max_w)))

    # ------------------------------------------------------------------
    # Nhận diện & Thích nghi với Vùng Hotspot / Khung giờ Surge
    # ------------------------------------------------------------------
    def _update_hotspot_beliefs(self, orders: Dict[int, Order], t: int) -> None:
        # Hỗ trợ Phase 2: Nếu file cấu hình mở có sẵn tham số chính thức
        static_hotspots = self.cfg.get("hotspots", [])
        surge_windows = self.cfg.get("surge_windows", [])
        
        in_surge = False
        for ts, te in surge_windows:
            if ts <= t <= te:
                in_surge = True
                break
                
        if static_hotspots and in_surge:
            self._dynamic_hotspots = {tuple(pos): 5.0 for pos in static_hotspots}
            return

        # Hỗ trợ Phase 1: Thích nghi mù bằng cách tự đếm mật độ đơn hàng thực tế trên bản đồ
        self._dynamic_hotspots.clear()
        for order in orders.values():
            if not order.picked and not order.delivered:
                pos = (order.sx, order.sy)
                self._dynamic_hotspots[pos] = self._dynamic_hotspots.get(pos, 0.0) + 1.0

    def _get_hotspot_bonus(self, pos: Position) -> float:
        # Thưởng điểm heuristic cho các ô nằm trong phạm vi khoảng cách Manhattan <= 3 với tâm Hotspot
        bonus = 0.0
        for h_pos, weight in self._dynamic_hotspots.items():
            dist = abs(pos[0] - h_pos[0]) + abs(pos[1] - h_pos[1])
            if dist <= 3:
                bonus += weight / (1.0 + dist)
        return min(3.0, bonus)

    # ------------------------------------------------------------------
    # ACO Core: Quản lý Pheromone & Heuristics định tuyến
    # ------------------------------------------------------------------
    def _pheromone_key(self, task_type: str, order: Order) -> Tuple[str, int]:
        return task_type, order.id

    def _get_pheromone(self, task_type: str, order: Order) -> float:
        return self._pheromone.get(self._pheromone_key(task_type, order), 1.0)

    def _evaporate_pheromone(self, active_order_ids: Set[int]) -> None:
        for key in list(self._pheromone.keys()):
            _, order_id = key
            if order_id not in active_order_ids:
                self._pheromone.pop(key, None)
                continue
            value = max(self._min_pheromone, self._pheromone[key] * (1.0 - self._rho))
            if math.isclose(value, self._min_pheromone):
                self._pheromone.pop(key, None)
            else:
                self._pheromone[key] = value

    def _reinforce(self, task_type: str, order: Order, amount: float) -> None:
        key = self._pheromone_key(task_type, order)
        current = self._pheromone.get(key, 1.0)
        self._pheromone[key] = min(self._max_pheromone, current + amount)

    def _aco_value(self, pheromone: float, heuristic: float) -> float:
        return (pheromone**self._alpha) * (max(heuristic, 1e-6) ** self._beta)

    def _delivery_heuristic(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> float:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        if distance >= INF: return 0.0

        T_max = int(self.cfg.get("T", 240))
        est_reward = delivery_reward(order, t + distance, T_max)
        
        current_w = sum(orders[oid].w for oid in shipper.bag if oid in orders)
        travel_cost = self._estimated_travel_cost(distance, current_w, shipper.W_max)
        
        net_profit = est_reward - travel_cost
        return max(0.01, net_profit) / (1.0 + distance)

    def _pickup_heuristic(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> float:
        pickup_pos = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)
        
        d1 = self._distance(shipper.position, pickup_pos)
        d2 = self._distance(pickup_pos, delivery_pos)
        if d1 >= INF or d2 >= INF: return 0.0

        T_max = int(self.cfg.get("T", 240))
        # Đơn hàng sẽ được hoàn thành tại thời điểm: t + d1 (đi đến chỗ lấy) + 1 (bước nhặt hàng) + d2 (đi giao)
        finish_t = t + d1 + 1 + d2
        est_reward = delivery_reward(order, finish_t, T_max)
        
        current_w = sum(orders[oid].w for oid in shipper.bag if oid in orders)
        
        # SỬA LỖI CHI PHÍ TẢI TRỌNG PHÂN TẦNG CHÍNH XÁC:
        # Đoạn d1: Mang khối lượng cũ trong túi
        cost_d1 = self._estimated_travel_cost(d1, current_w, shipper.W_max)
        # Bước đứng yên phát lệnh nhặt (tốn 1 tick hành động): Mang khối lượng cũ
        cost_pickup_step = self._estimated_travel_cost(1, current_w, shipper.W_max)
        # Đoạn d2: Túi đã được cộng thêm khối lượng của đơn hàng mới nhặt
        cost_d2 = self._estimated_travel_cost(d2, current_w + order.w, shipper.W_max)
        
        total_cost = cost_d1 + cost_pickup_step + cost_d2
        hotspot_bonus = self._get_hotspot_bonus(pickup_pos)
        
        net_profit = est_reward - total_cost + hotspot_bonus
        return max(0.01, net_profit) / (1.0 + d1 + 0.4 * d2)

    def _delivery_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, float, int]:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        T_max = int(self.cfg.get("T", 240))
        finish_t = t + distance
        is_late = 1 if finish_t > order.et else 0
        est_reward = delivery_reward(order, finish_t, T_max)
        return (is_late, -order.p, -est_reward, order.id)

    def _pickup_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, float, int]:
        pickup_pos = (order.sx, order.sy)
        d1 = self._distance(shipper.position, pickup_pos)
        d2 = self._distance(pickup_pos, (order.ex, order.ey))
        T_max = int(self.cfg.get("T", 240))
        finish_t = t + d1 + 1 + d2
        is_late = 1 if finish_t > order.et else 0
        return (is_late, -order.p, d1, -delivery_reward(order, finish_t, T_max), order.id)

    # ------------------------------------------------------------------
    # Phối hợp điều phối phân bổ đơn hàng cho Shipper
    # ------------------------------------------------------------------
    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Order]:
        candidates = [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]
        if not candidates: return None
        return min(candidates, key=lambda o: self._delivery_rank(shipper, o, t))

    def _select_pickup(self, shipper: Shipper, orders: Dict[int, Order], reserved_order_ids: Set[int], t: int) -> Optional[Order]:
        candidates = []
        for order in orders.values():
            if order.id in reserved_order_ids or order.picked or order.delivered: continue
            if not shipper.can_carry(order, orders): continue
            if self._pickup_heuristic(shipper, order, orders, t) <= 0.0: continue
            candidates.append(order)
        if not candidates: return None
        return min(candidates, key=lambda o: self._pickup_rank(shipper, o, t))

    def _assign_pickups(self, shippers: List[Shipper], orders: Dict[int, Order], t: int) -> Dict[int, Order]:
        candidates = []
        for shipper in shippers:
            for order in orders.values():
                if order.picked or order.delivered: continue
                if not shipper.can_carry(order, orders): continue
                h_val = self._pickup_heuristic(shipper, order, orders, t)
                if h_val <= 0.0: continue
                
                score = self._aco_value(self._get_pheromone("pickup", order), h_val)
                candidates.append((self._pickup_rank(shipper, order, t), -score, shipper.id, order))

        # Sắp xếp ưu tiên tối đa tính hợp lý của bầy kiến
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3].id))

        assignments: Dict[int, Order] = {}
        used_shippers: Set[int] = set()
        used_orders: Set[int] = set()

        for _, _, shipper_id, order in candidates:
            if shipper_id in used_shippers or order.id in used_orders: continue
            used_shippers.add(shipper_id)
            used_orders.add(order.id)
            assignments[shipper_id] = order

        return assignments

    def _should_pickup_before_delivery(self, shipper: Shipper, pickup: Order, delivery: Order, orders: Dict[int, Order], t: int) -> bool:
        if len(shipper.bag) >= shipper.K_max: return False
        
        d_direct = self._distance(shipper.position, (delivery.ex, delivery.ey))
        d_detour = self._distance(shipper.position, (pickup.sx, pickup.sy)) + 1 + self._distance((pickup.sx, pickup.sy), (delivery.ex, delivery.ey))
        
        if d_detour >= INF: return False
        if t + d_direct <= delivery.et < t + d_detour: return False
        if pickup.p > delivery.p: return True
        
        p_score = self._aco_value(self._get_pheromone("pickup", pickup), self._pickup_heuristic(shipper, pickup, orders, t))
        d_score = self._aco_value(self._get_pheromone("deliver", delivery), self._delivery_heuristic(shipper, delivery, orders, t))
        return p_score > d_score * 1.4

    def _find_alternative_move(self, current: Position, blocked_positions: Set[Position]) -> Move:
        for move in MOVES:
            nxt = valid_next_pos(current, move, self.grid)
            if nxt != current and nxt not in blocked_positions:
                return move
        return "S"

    # ------------------------------------------------------------------
    # Quy trình ra quyết định chuẩn hóa theo Vận hành Đề bài
    # ------------------------------------------------------------------
    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        T_max = int(self.cfg.get("T", 240))

        self._update_hotspot_beliefs(orders, t)

        delivery_orders = {shipper.id: self._select_delivery(shipper, orders, t) for shipper in shippers}
        idle_shippers = [shipper for shipper in shippers if delivery_orders[shipper.id] is None]

        reserved_pickups: Set[int] = set()
        assigned_pickups = self._assign_pickups(idle_shippers, orders, t)
        reserved_pickups.update(order.id for order in assigned_pickups.values())

        actions: Dict[int, Action] = {}
        occupied_next_positions: Set[Position] = set()

        # DUYỆT THEO THỨ TỰ ID TĂNG DẦN (Bắt buộc để áp dụng đúng luật giữ ô ưu tiên của đề bài)
        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = delivery_orders[shipper.id]
            pickup_order = assigned_pickups.get(shipper.id)

            # Kiểm tra xem có nên rẽ ngang nhặt thêm đơn không
            if delivery_order is not None:
                potential_pickup = self._select_pickup(shipper, orders, reserved_pickups, t)
                if potential_pickup is not None and self._should_pickup_before_delivery(shipper, potential_pickup, delivery_order, orders, t):
                    pickup_order = potential_pickup
                    reserved_pickups.add(pickup_order.id)

            move = "S"
            cargo_op = 0

            # Xử lý logic Thao tác Hàng hóa chuẩn hóa theo Gym API
            if pickup_order is not None:
                target_pos = (pickup_order.sx, pickup_order.sy)
                if shipper.position == target_pos:
                    move = "S"
                    cargo_op = 1  # Đứng yên tại điểm lấy hàng và phát lệnh nhặt hàng
                else:
                    move = self._next_move(shipper.position, target_pos)
                    cargo_op = 0
                
                f_t = t + self._distance(shipper.position, target_pos) + 1 + self._distance(target_pos, (pickup_order.ex, pickup_order.ey))
                self._reinforce("pickup", pickup_order, delivery_reward(pickup_order, f_t, T_max) * 0.1)

            elif delivery_order is not None:
                target_pos = (delivery_order.ex, delivery_order.ey)
                if shipper.position == target_pos:
                    move = "S"
                    cargo_op = 2  # Đứng yên tại điểm giao hàng và phát lệnh giao hàng
                else:
                    move = self._next_move(shipper.position, target_pos)
                    cargo_op = 0
                
                f_t = t + self._distance(shipper.position, target_pos)
                self._reinforce("deliver", delivery_order, delivery_reward(delivery_order, f_t, T_max) * 0.1)

            # Dự kiến vị trí bước tiếp theo
            next_pos = valid_next_pos(shipper.position, move, self.grid)

            # [CƠ CHẾ TRÁNH VA CHẠM]: Nếu ô dự định đi tiếp trùng với ô của một shipper ID nhỏ hơn đã xí trước
            if next_pos in occupied_next_positions and move != "S":
                # Tìm đường lánh nạn vòng quanh ô trống gần nhất, hủy bỏ lệnh nhặt/giao ở bước né này
                move = self._find_alternative_move(shipper.position, occupied_next_positions)
                next_pos = valid_next_pos(shipper.position, move, self.grid)
                cargo_op = 0

            occupied_next_positions.add(next_pos)
            actions[shipper.id] = (move, cargo_op)

        return actions

    # ------------------------------------------------------------------
    # Vòng lặp chấm điểm chính thức của hệ thống mô phỏng
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            active_order_ids = set(obs["orders"].keys())
            self._evaporate_pheromone(active_order_ids)
            
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )