from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.hotspot_tracker import HotspotTracker
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
PRINT_MAP_EVERY = 20
WRITE_MAP_TO_FILE = False
PRINT_MAP_TO_TERMINAL = False


def format_map(obs: dict) -> str:
    """Fallback map formatter để solver không phụ thuộc file phụ khi chạy Kaggle."""
    grid = obs["grid"]
    orders = obs["orders"]
    shippers = obs["shippers"]

    board = [["#" if cell == 1 else "." for cell in row] for row in grid]
    for order in orders.values():
        if not order.delivered:
            board[order.ex][order.ey] = "D"
        if not order.picked:
            board[order.sx][order.sy] = "P"

    for shipper in shippers:
        r, c = shipper.position
        board[r][c] = str(shipper.id) if shipper.id < 10 else chr(ord("A") + (shipper.id - 10) % 26)

    active_orders = len(orders)
    carried = sum(len(shipper.bag) for shipper in shippers)
    lines = [f"--- MAP t={obs['t']}/{obs['T']} active={active_orders} carried={carried} ---"]
    lines.extend(" ".join(row) for row in board)
    return "\n".join(lines)


def map_log_path(cfg: dict, prefix: str = "map") -> str:
    """Fallback log path; chỉ dùng khi WRITE_MAP_TO_FILE=True."""
    config_name = str(cfg.get("name", "unknown"))
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in config_name)
    return f"{prefix}_{safe_name}.txt"


class GreedyBFS(Solver):
    """
    Greedy BFS baseline cho Online MAPD.

    Solver chỉ cài phần policy:
    - chọn đơn cần giao/nhặt;
    - tìm đường bằng BFS trên grid hiện tại.
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.cfg = {"N": env.N, "C": env.C, "G": env.G, "T": env.T, "name": "unknown"}
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._hotspot_tracker = HotspotTracker(window=80, radius=3, max_hotspots=3)
        self._anchor_cache: Dict[int, Position] = {}

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        """Liệt kê các ô kề hợp lệ bằng valid_next_pos() của env."""
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        """Chạy BFS và lưu parent để lấy khoảng cách/next move."""
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {
            start: (None, "S")
        }

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

    def _distance(self, start: Position, goal: Position) -> int:
        """
        Khoảng cách đường đi ngắn nhất trên grid có vật cản.
        """
        if start == goal:
            return 0

        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        distance = 0
        current = goal
        while current != start:
            previous, _ = parent[current]
            if previous is None:
                self._distance_cache[key] = INF
                return INF
            current = previous
            distance += 1

        self._distance_cache[key] = distance
        return distance

    def _next_move(self, start: Position, goal: Position) -> Move:
        """Bước đi đầu tiên trên đường BFS từ start tới goal."""
        if start == goal:
            return "S"

        key = (start, goal)
        if key in self._next_move_cache:
            return self._next_move_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        current = goal
        while True:
            previous, move = parent[current]
            if previous is None:
                self._next_move_cache[key] = "S"
                return "S"
            if previous == start:
                self._next_move_cache[key] = move
                return move
            current = previous

    # ------------------------------------------------------------------
    # Policy: chọn đơn
    # ------------------------------------------------------------------
    def _estimated_reward(self, order: Order, finish_t: int) -> float:
        """Ước lượng reward nếu order được giao tại finish_t theo đúng hàm điểm của env."""
        return delivery_reward(order, finish_t, int(self.cfg.get("T", 1)))

    def _delivery_key(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int]:
        """Ưu tiên đơn đang mang có khả năng giao đúng hạn, deadline gấp, priority cao."""
        distance = self._distance(shipper.position, (order.ex, order.ey))
        finish_t = t + distance
        slack = order.et - finish_t
        return (
            1 if finish_t > order.et else 0,
            slack,
            distance,
            -order.p,
            order.id,
        )

    def _map_scale(self) -> int:
        """Kích thước cạnh map; dùng max(1, ...) để tránh chia cho 0 ở các heuristic."""
        return max(1, int(self.cfg.get("N", len(self.grid))))

    def _map_stats(self) -> Tuple[int, int, float, float]:
        """
        Trích xuất vài đặc trưng tổng quát của map.

        Các đặc trưng này không phụ thuộc tên config cụ thể nào mà chỉ dựa trên grid và số shipper, để các 
          heuristic sau có thể tự điều chỉnh theo cỡ map và tình trạng vật cản:
        - n: kích thước cạnh map;
        - c: số shipper;
        - obstacle_ratio: map càng nhiều vật cản thì khả năng có bottleneck càng cao;
        - free_per_shipper: số ô trống trung bình mỗi shipper phải cover.
        """
        n = self._map_scale()
        c = max(1, int(self.cfg.get("C", 1)))
        total = n * n
        blocked = sum(cell == 1 for row in self.grid for cell in row)
        obstacle_ratio = blocked / max(total, 1)
        free_per_shipper = (total - blocked) / c
        return n, c, obstacle_ratio, free_per_shipper

    def _map_mode(self) -> str:
        """Phân loại thô để mô tả map trong code/report, không dùng hard-code theo config."""
        n, _, _, _ = self._map_stats()
        if n <= 10:
            return "small"
        if n < 18:
            return "medium"
        return "large"

    def _should_use_region_policy(self) -> bool:
        """
        Quyết định có nên phạt đơn khác vùng hay không.

        Region policy chỉ nên bật khi shipper dễ mất nhiều thời gian chạy xuyên map:
        map lớn, ít shipper tương đối, nhiều vật cản, hoặc mỗi shipper phải cover
        một vùng trống lớn. Điều kiện này giữ C5 linh hoạt nhưng giúp C6 bớt
        tranh đơn quá xa qua bottleneck.
        """
        n, c, obstacle_ratio, free_per_shipper = self._map_stats()
        if n >= 30 and c <= 3:
            return False
        return (
            n >= 20
            or (n >= 18 and c <= 4 and obstacle_ratio >= 0.22)
            or (free_per_shipper >= 60 and obstacle_ratio >= 0.22)
        )

    def _region_weight(self) -> float:
        """
        Trọng số phạt khác vùng.

        Đây là soft penalty chứ không cấm tuyệt đối. Đơn khác vùng vẫn có thể
        được chọn nếu reward/time đủ tốt, nhưng sẽ kém hấp dẫn hơn đơn cùng vùng.
        """
        if not self._should_use_region_policy():
            return 0.0

        n, c, obstacle_ratio, free_per_shipper = self._map_stats()
        weight = 0.25
        if n >= 20:
            weight += 0.20
        if c <= 4:
            weight += 0.10
        if obstacle_ratio >= 0.25:
            weight += 0.10
        if free_per_shipper >= 70:
            weight += 0.10
        return min(weight, 0.65)

    def _region_of(self, pos: Position) -> int:
        """
        Gán một ô vào vùng hoạt động.

        Khi region policy bật, chia map thành 4 phần tư. Khi không bật, chỉ chia
        trái/phải nhẹ để hàm vẫn có định nghĩa nhưng penalty bằng 0.
        """
        r, c = pos
        n = self._map_scale()
        if self._should_use_region_policy():
            return (0 if r < n // 2 else 2) + (0 if c < n // 2 else 1)
        return 0 if c < n // 2 else 1

    def _region_penalty(self, shipper: Shipper, order: Order) -> float:
        """
        Phạt mềm khi pickup/dropoff nằm khác vùng hiện tại của shipper.

        Pickup khác vùng bị phạt mạnh hơn dropoff khác vùng vì chạy tới điểm lấy
        hàng xa thường là nguyên nhân chính làm shipper bỏ lỡ nhiều đơn gần.
        """
        weight = self._region_weight()
        if weight <= 0.0:
            return 0.0

        shipper_region = self._region_of(shipper.position)
        pickup_region = self._region_of((order.sx, order.sy))
        delivery_region = self._region_of((order.ex, order.ey))

        penalty = 0
        if pickup_region != shipper_region:
            penalty += 8
        if delivery_region != shipper_region:
            penalty += 4
        return weight * penalty

    def _hotspot_bonus(self, order: Order) -> float:
        """
        Bonus nhẹ cho đơn có pickup gần cụm đơn mới xuất hiện gần đây.

        Đây là hotspot detection online: chỉ dựa trên observation đã reveal,
        không dùng thông tin hotspot ẩn của môi trường.
        """
        if not self._should_use_region_policy():
            return 0.0
        return self._hotspot_tracker.score((order.sx, order.sy))

    def _use_local_policy(self, obs: dict) -> bool:
        """
        Chọn policy đơn giản khi workload còn nhẹ so với số shipper.

        Tiêu chí này phụ thuộc trạng thái hiện tại, không phụ thuộc tên config
        hay một cỡ map cố định.
        """
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        active_orders = len(orders)
        carried = sum(len(shipper.bag) for shipper in shippers)
        pressure = active_orders + carried
        n = self._map_scale()
        if n <= 10:
            return pressure <= max(2, 2 * len(shippers))
        if n <= 12 or self._should_use_region_policy():
            return pressure <= max(2, len(shippers))
        return pressure <= max(2, 2 * len(shippers))

    def _use_distance_pruning(self, obs: dict) -> bool:
        """Bật lọc xa/trễ cùng điều kiện với region policy để tránh over-pruning map nhỏ."""
        return self._should_use_region_policy()

    def _pickup_key(self, shipper: Shipper, order: Order, t: int, prefer_near: bool = False) -> Tuple[float, ...]:
        """
        Khóa sắp xếp để chọn pickup.

        Nếu prefer_near=True, dùng policy local: ưu tiên đơn gần trước, phù hợp
        khi workload nhẹ. Ngược lại dùng policy global: ưu tiên reward mỗi bước,
        sau đó region penalty, deadline, priority và tổng quãng đường.
        """
        pickup_pos = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup_pos)
        delivery_distance = self._distance(pickup_pos, delivery_pos)
        finish_t = t + pickup_distance + delivery_distance
        slack = order.et - finish_t
        route_distance = pickup_distance + delivery_distance
        if prefer_near:
            # Local mode: đơn gần thường tốt hơn vì chưa có nhiều việc để tối ưu global.
            return (
                1 if finish_t > order.et else 0,
                0.0,
                -self._hotspot_bonus(order),
                pickup_distance,
                -order.p,
                float(slack),
                delivery_distance,
                order.id,
            )

        value_per_step = self._estimated_reward(order, finish_t) / (route_distance + 1)
        hotspot_boost = 1.0 + 0.12 * self._hotspot_bonus(order)
        # Global mode: score chính là reward/time; region chỉ là phạt mềm sau đó.
        return (
            1 if finish_t > order.et else 0,
            -(value_per_step * hotspot_boost),
            self._region_penalty(shipper, order),
            slack,
            -order.p,
            route_distance,
            order.id,
        )

    def _can_consider_pickup(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        use_pruning: bool = False,
    ) -> bool:
        """
        Kiểm tra order có đáng đưa vào danh sách ứng viên pickup không.

        Luôn loại đơn không chở được hoặc không có đường đi. Khi use_pruning=True,
        loại thêm các đơn ưu tiên thấp mà dự kiến vừa xa vừa trễ, vì đuổi theo
        chúng thường làm shipper mất cơ hội xử lý cụm đơn gần hơn.
        """
        if not shipper.can_carry(order, orders):
            return False

        pickup_pos = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup_pos)
        delivery_distance = self._distance(pickup_pos, delivery_pos)
        if pickup_distance >= INF or delivery_distance >= INF:
            return False
        if not use_pruning:
            return True

        map_scale = self._map_scale()
        finish_t = t + pickup_distance + delivery_distance
        lateness = max(0, finish_t - order.et)
        route_distance = pickup_distance + delivery_distance
        value_per_step = self._estimated_reward(order, finish_t) / (route_distance + 1)

        # Các ngưỡng này chỉ áp dụng khi map có dấu hiệu rộng/bottleneck.
        # Đơn priority cao vẫn được giữ lại nhiều hơn vì reward trễ vẫn đáng kể.
        if order.p == 1 and lateness > max(5, map_scale // 2):
            return False
        if order.p <= 2 and pickup_distance > max(6, map_scale // 2) and lateness > 0:
            return False
        if order.p == 1 and pickup_distance > max(8, (2 * map_scale) // 3):
            return False
        if value_per_step < 0.08 and lateness > 0:
            return False

        return True

    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Order]:
        """
        Chọn đơn đang mang để đi giao.
        """
        carried_orders = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried_orders:
            return None

        return min(
            carried_orders,
            key=lambda order: self._delivery_key(shipper, order, t),
        )

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
        t: int,
        prefer_near: bool = False,
        use_pruning: bool = False,
    ) -> Optional[Order]:
        """Chọn đơn chưa nhặt phù hợp nhất với vị trí và năng lực shipper."""
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not self._can_consider_pickup(shipper, order, orders, t, use_pruning):
                continue
            candidates.append(order)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda order: self._pickup_key(shipper, order, t, prefer_near),
        )

    def _should_pickup_before_delivery(
        self,
        shipper: Shipper,
        pickup_order: Order,
        delivery_order: Order,
        t: int,
    ) -> bool:
        """
        Quyết định có nên pickup thêm trước khi giao đơn đang mang.

        Chỉ cho phép nếu còn đủ sức chứa, detour không làm đơn đang đúng hạn bị
        trễ, và phần detour nằm trong giới hạn thích nghi theo cỡ map/khoảng cách
        giao hiện tại. Điều này giúp gom đơn ở C4-C5 nhưng không làm shipper ôm
        quá nhiều hàng.
        """
        map_scale = self._map_scale()
        carried_ratio = len(shipper.bag) / max(shipper.K_max, 1)
        if carried_ratio >= 0.75:
            return False
        if len(shipper.bag) >= shipper.K_max:
            return False

        current_delivery_distance = self._distance(shipper.position, (delivery_order.ex, delivery_order.ey))
        direct_finish = t + current_delivery_distance

        pickup_pos = (pickup_order.sx, pickup_order.sy)
        delivery_pos = (delivery_order.ex, delivery_order.ey)
        detour_finish = (
            t
            + self._distance(shipper.position, pickup_pos)
            + self._distance(pickup_pos, delivery_pos)
        )
        if detour_finish >= INF:
            return False

        direct_on_time = direct_finish <= delivery_order.et
        detour_on_time = detour_finish <= delivery_order.et
        if direct_on_time and not detour_on_time:
            return False

        detour_extra = detour_finish - direct_finish
        pickup_score = self._pickup_key(shipper, pickup_order, t)
        delivery_score = self._delivery_key(shipper, delivery_order, t)
        adaptive_limit = max(3, min(map_scale // 3, delivery_score[2] // 2))

        return (
            detour_extra <= adaptive_limit
            and pickup_score[0] == 0
            and (
                pickup_order.p > delivery_order.p
                or -pickup_score[4] >= self._estimated_reward(delivery_order, direct_finish) / (delivery_score[2] + 1)
            )
        )

    # ------------------------------------------------------------------
    # Policy: tạo action
    # ------------------------------------------------------------------
    def _move_towards(
        self,
        shipper: Shipper,
        goal: Position,
        reserved_next_positions: Optional[set[Position]] = None,
    ) -> Tuple[Move, Position]:
        """
        Lấy bước đi kế tiếp và vị trí dự kiến sau bước đó.

        reserved_next_positions chứa các ô đã được shipper xử lý trước đó chọn
        trong cùng timestep. Nếu đường BFS tốt nhất đi vào ô đã bị giữ, thử các
        bước thay thế và chọn bước còn hợp lệ gần goal nhất. Đây là tránh va
        chạm một bước, không phải multi-agent path planning đầy đủ.
        """
        reserved_next_positions = reserved_next_positions or set()
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        if next_position not in reserved_next_positions:
            return move, next_position

        # Bước chính bị trùng ô; thử đứng yên hoặc rẽ hướng khác để giảm kẹt.
        alternatives: List[Tuple[int, Move, Position]] = []
        for alt_move in ("S",) + MOVES:
            alt_position = valid_next_pos(shipper.position, alt_move, self.grid)
            if alt_position in reserved_next_positions:
                continue
            alternatives.append((self._distance(alt_position, goal), alt_move, alt_position))

        if not alternatives:
            return "S", shipper.position
        _, best_move, best_position = min(alternatives)
        return best_move, best_position

    def _delivery_action(
        self,
        shipper: Shipper,
        order: Order,
        reserved_next_positions: Optional[set[Position]] = None,
    ) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal, reserved_next_positions)

        # Với env chuẩn, op=2 nghĩa là giao tất cả đơn trong bag
        # có đích tại ô hiện tại sau khi di chuyển.
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(
        self,
        shipper: Shipper,
        order: Order,
        reserved_next_positions: Optional[set[Position]] = None,
    ) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal, reserved_next_positions)

        # cargo_op = 1: env/Shipper.pickup_best() sẽ nhặt một đơn tốt nhất tại ô hiện tại.
        return (move, 1) if next_position == goal else (move, 0)

    def _next_position_from_action(self, shipper: Shipper, action: Action) -> Position:
        """Tính ô mà shipper sẽ chiếm sau action, dùng để reserve trong cùng timestep."""
        move, _ = action
        return valid_next_pos(shipper.position, move, self.grid)

    def _nearest_valid_to(self, target: Position) -> Position:
        """Tìm ô trống gần target nhất, dùng cho anchor vùng khi target rơi vào vật cản."""
        if is_valid_cell(target, self.grid):
            return target

        best: Tuple[int, Position] = (INF, target)
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell != 0:
                    continue
                distance = abs(r - target[0]) + abs(c - target[1])
                if distance < best[0]:
                    best = (distance, (r, c))
        return best[1]

    def _region_anchor(self, region: int) -> Position:
        """Anchor đơn giản cho từng vùng; chỉ dùng khi shipper rảnh trên map vừa/lớn."""
        if region in self._anchor_cache:
            return self._anchor_cache[region]

        n = self._map_scale()
        anchors = {
            0: (n // 4, n // 4),
            1: (n // 4, (3 * n) // 4),
            2: ((3 * n) // 4, n // 4),
            3: ((3 * n) // 4, (3 * n) // 4),
        }
        anchor = self._nearest_valid_to(anchors.get(region, (n // 2, n // 2)))
        self._anchor_cache[region] = anchor
        return anchor

    def _idle_goal(self, shipper: Shipper, obs: dict) -> Optional[Position]:
        """
        Vị trí chờ cho shipper rảnh.

        Với config T lớn, đơn xuất hiện thưa hơn. Nếu shipper đứng yên ở vị trí xấu,
        nó dễ mất nhiều bước khi đơn mới reveal. Heuristic này chỉ dùng dữ liệu
        public: hotspot online từ đơn đã reveal và anchor theo hình học map.
        """
        n = self._map_scale()
        if n < 12:
            return None

        hotspots = self._hotspot_tracker.hotspots()
        if hotspots:
            return min(hotspots, key=lambda pos: self._distance(shipper.position, pos))

        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        carried = sum(len(s.bag) for s in shippers)
        pressure = len(orders) + carried
        if pressure > len(shippers):
            return None

        return self._region_anchor(self._region_of(shipper.position))

    def _idle_action(
        self,
        shipper: Shipper,
        obs: dict,
        reserved_next_positions: Optional[set[Position]] = None,
    ) -> Action:
        goal = self._idle_goal(shipper, obs)
        if goal is None or goal == shipper.position:
            return ("S", 0)
        move, _ = self._move_towards(shipper, goal, reserved_next_positions)
        return (move, 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        """
        Tạo action cho toàn bộ shipper ở timestep hiện tại.

        Luồng chính:
        1. Chọn delivery tốt nhất cho từng shipper đang mang hàng.
        2. Nếu workload nhẹ, dùng local policy để phản ứng nhanh với đơn gần.
        3. Nếu workload cao, tạo tất cả cặp (shipper, order), sắp xếp theo
           _pickup_key rồi greedy matching để tránh nhiều shipper giành cùng đơn.
        4. Khi xuất action, reserve ô kế tiếp để giảm va chạm trực tiếp.
        """
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        self._hotspot_tracker.update(obs)

        actions: Dict[int, Action] = {}
        assigned_pickups: Dict[int, Order] = {}
        used_order_ids: set[int] = set()
        delivery_by_shipper = {
            shipper.id: self._select_delivery(shipper, orders, t)
            for shipper in shippers
        }
        prefer_local = self._use_local_policy(obs)
        use_pruning = self._use_distance_pruning(obs)

        if prefer_local:
            # Workload nhẹ: overhead global matching không đáng, shipper tự chọn đơn gần.
            # Reserve ô kế tiếp chỉ khi map đủ nhỏ hoặc region policy đang bật.
            # Với map rộng nhưng nhiều shipper kiểu C5, reserve quá sớm có thể làm giảm throughput.
            reserved_pickups: set[int] = set()
            reserved_next_positions: set[Position] = set()
            use_local_reserve = self._map_scale() < 18 or self._should_use_region_policy()
            action_order = sorted(
                shippers,
                key=lambda s: (
                    self._delivery_key(s, delivery_by_shipper[s.id], t)
                    if delivery_by_shipper[s.id] is not None
                    else (1, INF, INF, 0, s.id)
                ),
            )

            for shipper in action_order:
                delivery_order = delivery_by_shipper[shipper.id]
                pickup_order = self._select_pickup(
                    shipper,
                    orders,
                    reserved_pickups,
                    t,
                    prefer_near=True,
                    use_pruning=use_pruning,
                )

                if delivery_order is not None:
                    if (
                        pickup_order is not None
                        and self._should_pickup_before_delivery(shipper, pickup_order, delivery_order, t)
                    ):
                        reserved_pickups.add(pickup_order.id)
                        reserve = reserved_next_positions if use_local_reserve else None
                        action = self._pickup_action(shipper, pickup_order, reserve)
                        actions[shipper.id] = action
                        if use_local_reserve:
                            reserved_next_positions.add(self._next_position_from_action(shipper, action))
                        continue
                    reserve = reserved_next_positions if use_local_reserve else None
                    action = self._delivery_action(shipper, delivery_order, reserve)
                    actions[shipper.id] = action
                    if use_local_reserve:
                        reserved_next_positions.add(self._next_position_from_action(shipper, action))
                    continue

                if pickup_order is not None:
                    reserved_pickups.add(pickup_order.id)
                    reserve = reserved_next_positions if use_local_reserve else None
                    action = self._pickup_action(shipper, pickup_order, reserve)
                    actions[shipper.id] = action
                    if use_local_reserve:
                        reserved_next_positions.add(self._next_position_from_action(shipper, action))
                    continue

                reserve = reserved_next_positions if use_local_reserve else None
                action = self._idle_action(shipper, obs, reserve)
                actions[shipper.id] = action
                if use_local_reserve:
                    reserved_next_positions.add(self._next_position_from_action(shipper, action))
            return actions

        # Workload cao: tạo matching toàn cục shipper-order để shipper gần/phù hợp
        # hơn được quyền lấy đơn trước, thay vì duyệt đơn theo id shipper.
        pickup_pairs: List[Tuple[Tuple[float, ...], int, Order]] = []
        for shipper in shippers:
            delivery_order = delivery_by_shipper[shipper.id]
            for order in orders.values():
                if not self._can_consider_pickup(shipper, order, orders, t, use_pruning):
                    continue
                if delivery_order is not None and not self._should_pickup_before_delivery(shipper, order, delivery_order, t):
                    continue
                pickup_pairs.append((self._pickup_key(shipper, order, t, prefer_near=False), shipper.id, order))

        for _, shipper_id, order in sorted(pickup_pairs):
            if shipper_id in assigned_pickups or order.id in used_order_ids:
                continue
            assigned_pickups[shipper_id] = order
            used_order_ids.add(order.id)

        # Shipper có delivery gấp được quyết định action trước để giữ đường.
        reserved_next_positions: set[Position] = set()
        action_order = sorted(
            shippers,
            key=lambda s: (
                self._delivery_key(s, delivery_by_shipper[s.id], t)
                if delivery_by_shipper[s.id] is not None
                else (1, INF, INF, 0, s.id)
            ),
        )

        for shipper in action_order:
            pickup_order = assigned_pickups.get(shipper.id)
            delivery_order = delivery_by_shipper[shipper.id]

            if pickup_order is not None:
                action = self._pickup_action(shipper, pickup_order, reserved_next_positions)
            elif delivery_order is not None:
                action = self._delivery_action(shipper, delivery_order, reserved_next_positions)
            else:
                action = self._idle_action(shipper, obs, reserved_next_positions)

            actions[shipper.id] = action
            reserved_next_positions.add(self._next_position_from_action(shipper, action))

        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        self._hotspot_tracker.reset()
        map_log = None

        if WRITE_MAP_TO_FILE:
            map_log = open(map_log_path(self.cfg, prefix="greedy_bfs_map"), "w", encoding="utf-8")

        try:
            while not obs.get("done", False):
                if PRINT_MAP_EVERY > 0 and obs["t"] % PRINT_MAP_EVERY == 0:
                    map_text = format_map(obs)
                    if PRINT_MAP_TO_TERMINAL:
                        print("\n" + map_text)
                    if map_log is not None:
                        map_log.write(map_text + "\n\n")

                actions = self._decide_actions(obs)
                obs, _, done, _ = self.env.step(actions)
                if done:
                    break
        finally:
            if map_log is not None:
                map_log.close()

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
