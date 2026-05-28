from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
PRINT_MAP_EVERY = 20
WRITE_MAP_TO_FILE = True
PRINT_MAP_TO_TERMINAL = False


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
        self.cfg = {"N": env.N, "C": env.C, "G": env.G, "T": env.T, "name": env.config_name}
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}

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

    def _pickup_key(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int, int]:
        """Ước lượng lợi ích khi đi lấy: pickup gần, giao kịp hạn, priority cao."""
        pickup_pos = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup_pos)
        delivery_distance = self._distance(pickup_pos, delivery_pos)
        finish_t = t + pickup_distance + delivery_distance
        slack = order.et - finish_t
        return (
            1 if finish_t > order.et else 0,
            pickup_distance,
            -order.p,
            slack,
            delivery_distance,
            order.id,
        )

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
    ) -> Optional[Order]:
        """Chọn đơn chưa nhặt phù hợp nhất với vị trí và năng lực shipper."""
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            pickup_distance = self._distance(shipper.position, (order.sx, order.sy))
            delivery_distance = self._distance((order.sx, order.sy), (order.ex, order.ey))
            if pickup_distance >= INF or delivery_distance >= INF:
                continue
            candidates.append(order)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda order: self._pickup_key(shipper, order, t),
        )

    def _should_pickup_before_delivery(
        self,
        shipper: Shipper,
        pickup_order: Order,
        delivery_order: Order,
        t: int,
    ) -> bool:
        """Cho phép ghé lấy đơn gần nếu không làm đơn đang mang bị trễ rõ rệt."""
        if int(self.cfg.get("N", 0)) < 12:
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

        return (
            detour_extra <= 3
            and pickup_score[0] == 0
            and (
                pickup_order.p > delivery_order.p
                or pickup_score[1] <= max(2, delivery_score[2])
            )
        )

    # ------------------------------------------------------------------
    # Policy: tạo action
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        """
        Lấy bước đi kế tiếp và vị trí dự kiến sau bước đó.
        """
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return move, next_position

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)

        # Với env chuẩn, op=2 nghĩa là giao tất cả đơn trong bag
        # có đích tại ô hiện tại sau khi di chuyển.
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)

        # cargo_op = 1: env/Shipper.pickup_best() sẽ nhặt một đơn tốt nhất tại ô hiện tại.
        return (move, 1) if next_position == goal else (move, 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))

        actions: Dict[int, Action] = {}
        reserved_pickups: set[int] = set()

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = self._select_delivery(shipper, orders, t)
            pickup_order = self._select_pickup(shipper, orders, reserved_pickups, t)

            if delivery_order is not None:
                if (
                    pickup_order is not None
                    and self._should_pickup_before_delivery(shipper, pickup_order, delivery_order, t)
                ):
                    reserved_pickups.add(pickup_order.id)
                    actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                    continue
                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                continue

            actions[shipper.id] = ("S", 0)

        return actions

    # ------------------------------------------------------------------
    # Debug: in trạng thái map
    # ------------------------------------------------------------------
    def _format_map(self, obs: dict) -> str:
        """
        Tạo text mô tả trạng thái map hiện tại.

        Ký hiệu:
          #  vật cản
          .  ô trống
          P  điểm lấy hàng của đơn chưa nhặt
          D  điểm giao hàng của đơn chưa giao
          0..9 / A..Z  shipper id
        """
        grid = obs["grid"]
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        board = [["#" if cell == 1 else "." for cell in row] for row in grid]

        for order in orders.values():
            if not order.delivered:
                board[order.ex][order.ey] = "D"
            if not order.picked:
                board[order.sx][order.sy] = "P"

        for shipper in shippers:
            r, c = shipper.position
            if shipper.id < 10:
                marker = str(shipper.id)
            else:
                marker = chr(ord("A") + (shipper.id - 10) % 26)
            board[r][c] = marker

        active_orders = len(orders)
        carried = sum(len(shipper.bag) for shipper in shippers)
        lines = [
            f"--- MAP t={obs['t']}/{obs['T']} active={active_orders} carried={carried} ---"
        ]
        lines.extend(" ".join(row) for row in board)
        return "\n".join(lines)

    def _print_map(self, obs: dict) -> None:
        """In trạng thái map hiện tại ra terminal."""
        print("\n" + self._format_map(obs))

    def _map_log_path(self) -> str:
        """Tên file log map riêng cho từng config."""
        config_name = str(self.cfg.get("name", "unknown"))
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in config_name)
        return f"greedy_bfs_map_{safe_name}.txt"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        map_log = None

        if WRITE_MAP_TO_FILE:
            map_log = open(self._map_log_path(), "w", encoding="utf-8")

        try:
            while not obs.get("done", False):
                if PRINT_MAP_EVERY > 0 and obs["t"] % PRINT_MAP_EVERY == 0:
                    map_text = self._format_map(obs)
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
