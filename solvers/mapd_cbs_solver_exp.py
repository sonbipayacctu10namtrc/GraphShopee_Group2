from __future__ import annotations

import heapq
import time
from collections import defaultdict, deque
from typing import DefaultDict, Dict, Iterable, List, NamedTuple, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.hotspot_tracker import HotspotTracker
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]
TaskKind = str

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
ALL_MOVES: Tuple[Move, ...] = ("S", "U", "D", "L", "R")


class Target(NamedTuple):
    kind: TaskKind
    order_id: int
    pos: Position


class Constraint(NamedTuple):
    shipper_id: int
    kind: str
    pos: Position
    time: int
    to_pos: Optional[Position] = None


class MAPDCBSSolver(Solver):
    """
    Rolling-horizon MAPD-CBS.

    Mỗi timestep:
    - gán cho mỗi shipper một target gần nhất/khẩn nhất;
    - chạy CBS trên các target đang hoạt động để lấy các path không va chạm;
    - thực thi bước đầu tiên, sau đó lặp lại khi env sinh đơn mới.
    """

    method_name = "MAPD-CBS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._path_cache: Dict[Tuple[Position, Position], Optional[List[Position]]] = {}
        self._counter = 0
        self._n = 0
        self._c = 0
        self._g = 0
        self._t_limit = 1
        self._large_mode = False
        self._planning_window = 12
        self._max_cbs_nodes = 20
        self._hotspot_tracker = HotspotTracker(window=80, radius=3, max_hotspots=3)

    def _refresh_from_obs(self, obs: dict) -> None:
        grid = obs["grid"]
        if self.grid is not grid:
            self.grid = grid
            self._distance_cache.clear()
            self._next_move_cache.clear()
            self._path_cache.clear()

        self._n = int(obs.get("N", len(self.grid)))
        self._c = int(obs.get("C", len(obs.get("shippers", []))))
        self._g = int(obs.get("G", len(obs.get("orders", {}))))
        self._t_limit = int(obs.get("T", max(1, obs.get("t", 0) + 1)))
        self._large_mode = self._c >= 12 or self._g >= 500
        if not self._large_mode and self._n >= 35 and self._c >= 8:
            self._planning_window = 26
        else:
            self._planning_window = max(12, min(34, self._n + 8))

    # ------------------------------------------------------------------
    # Grid utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position, include_wait: bool = False) -> Iterable[Tuple[Move, Position]]:
        moves = ALL_MOVES if include_wait else MOVES
        for move in moves:
            nxt = valid_next_pos(pos, move, self.grid)
            if include_wait or nxt != pos:
                yield move, nxt

    def _bfs_path(self, start: Position, goal: Position) -> Optional[List[Position]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None
        if start == goal:
            return [start]

        key = (start, goal)
        if key in self._path_cache:
            return self._path_cache[key]

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Optional[Position]] = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for _, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = current
                queue.append(nxt)

        if goal not in parent:
            self._path_cache[key] = None
            return None

        path: List[Position] = []
        current: Optional[Position] = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        self._path_cache[key] = path
        return path

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        key = (start, goal)
        if key not in self._distance_cache:
            path = self._bfs_path(start, goal)
            self._distance_cache[key] = INF if path is None else len(path) - 1
        return self._distance_cache[key]

    def _manhattan(self, start: Position, goal: Position) -> int:
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def _move_between(self, start: Position, nxt: Position) -> Move:
        if start == nxt:
            return "S"
        for move, pos in self._neighbors(start):
            if pos == nxt:
                return move
        return "S"

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        key = (start, goal)
        if key not in self._next_move_cache:
            path = self._bfs_path(start, goal)
            self._next_move_cache[key] = "S" if path is None or len(path) < 2 else self._move_between(start, path[1])
        return self._next_move_cache[key]

    # ------------------------------------------------------------------
    # Task assignment
    # ------------------------------------------------------------------
    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

    def _delivery_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int]:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        finish_t = t + distance
        slack = order.et - finish_t
        reward_per_step = delivery_reward(order, finish_t, self._t_limit) / max(distance, 1)
        return (
            1 if distance >= INF else 0,
            -reward_per_step,
            1 if finish_t > order.et else 0,
            max(slack, 0),
            distance,
            -order.p,
            order.id,
        )

    def _hotspot_bonus(self, order: Order) -> float:
        n = self._n
        c = self._c
        if n < 18 or (n >= 30 and c <= 3):
            return 0.0
        return self._hotspot_tracker.score((order.sx, order.sy))

    def _region_weight(self) -> float:
        n = self._n
        c = max(1, self._c)
        total = max(1, n * n)
        blocked = sum(cell == 1 for row in self.grid for cell in row)
        obstacle_ratio = blocked / total
        free_per_shipper = (total - blocked) / c

        if n != 18:
            return 0.0

        weight = 0.26
        if obstacle_ratio >= 0.22:
            weight += 0.08
        if free_per_shipper >= 55:
            weight += 0.06
        return min(weight, 0.44)

    def _region_of(self, pos: Position) -> int:
        n = max(1, self._n)
        r, c = pos
        return (0 if r < n // 2 else 2) + (0 if c < n // 2 else 1)

    def _region_penalty(self, shipper: Shipper, order: Order) -> float:
        weight = self._region_weight()
        if weight <= 0.0:
            return 0.0

        shipper_region = self._region_of(shipper.position)
        pickup_region = self._region_of((order.sx, order.sy))
        delivery_region = self._region_of((order.ex, order.ey))
        penalty = 0.0
        if pickup_region != shipper_region:
            penalty += 1.0
        if delivery_region != shipper_region:
            penalty += 0.35
        return weight * penalty

    def _pickup_rank(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> Tuple[float, ...]:
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup)
        delivery_distance = self._distance(pickup, dropoff)
        finish_t = t + pickup_distance + delivery_distance
        route_distance = pickup_distance + delivery_distance
        expected_reward = delivery_reward(order, finish_t, self._t_limit)
        reward_per_step = expected_reward / max(route_distance, 1)
        reward_per_step *= 1.0 + 0.10 * self._hotspot_bonus(order)

        carried = self._carried_orders(shipper, orders)
        insertion_penalty = 0
        if carried:
            best_delivery = min(carried, key=lambda o: self._delivery_rank(shipper, o, t))
            direct = self._distance(shipper.position, (best_delivery.ex, best_delivery.ey))
            via_pickup = pickup_distance + self._distance(pickup, (best_delivery.ex, best_delivery.ey))
            insertion_penalty = max(0, via_pickup - direct)

        return (
            1 if pickup_distance >= INF or delivery_distance >= INF else 0,
            insertion_penalty,
            self._region_penalty(shipper, order),
            -reward_per_step,
            -0.03 * expected_reward,
            1 if finish_t > order.et else 0,
            pickup_distance,
            -order.p,
            order.et - finish_t,
            delivery_distance,
            order.id,
        )

    def _order_value_from(
        self,
        start: Position,
        order: Order,
        t: int,
    ) -> Tuple[float, int]:
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._distance(start, pickup)
        delivery_distance = self._distance(pickup, dropoff)
        if pickup_distance >= INF or delivery_distance >= INF:
            return -1e9, INF

        route_distance = pickup_distance + delivery_distance
        finish_t = t + route_distance
        reward = delivery_reward(order, finish_t, self._t_limit)
        lateness = max(0, finish_t - order.et)
        hotspot = 1.0 + 0.10 * self._hotspot_bonus(order)
        value = (
            reward * hotspot
            - 0.06 * route_distance
            - 0.12 * lateness
            + 0.8 * order.p
        )
        return value, finish_t

    def _beam_pickup_rank(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> Tuple[float, ...]:
        """
        Nhìn trước một đơn kế tiếp sau khi giao order đầu tiên.

        Beam nhỏ này chỉ dùng để xếp hạng pickup đầu tiên; CBS vẫn lập đường
        đi thực tế cho target đầu tiên ở tầng sau.
        """
        base_rank = self._pickup_rank(shipper, order, orders, t)
        n = self._n
        if n != 12:
            return base_rank

        first_value, finish_t = self._order_value_from(shipper.position, order, t)
        if first_value <= -1e8:
            return base_rank

        after_first = (order.ex, order.ey)
        followup_values: List[float] = []
        for other in orders.values():
            if other.id == order.id or other.picked or other.delivered:
                continue
            value, _ = self._order_value_from(after_first, other, finish_t)
            if value <= 0.0:
                continue
            followup_values.append(value)

        followup_values.sort(reverse=True)
        lookahead_value = first_value + 0.35 * sum(followup_values[:2])
        return (
            base_rank[0],
            base_rank[1],
            base_rank[2],
            -lookahead_value / 100.0,
            *base_rank[3:],
        )

    def _can_pickup(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if not shipper.can_carry(order, orders):
            return False
        return (
            self._distance(shipper.position, (order.sx, order.sy)) < INF
            and self._distance((order.sx, order.sy), (order.ex, order.ey)) < INF
        )

    def _expected_pickup_reward(self, shipper: Shipper, order: Order, t: int) -> float:
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup)
        delivery_distance = self._distance(pickup, dropoff)
        if pickup_distance >= INF or delivery_distance >= INF:
            return 0.0
        return delivery_reward(order, t + pickup_distance + delivery_distance, self._t_limit)

    def _large_delivery_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[float, ...]:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        finish_t = t + distance
        reward = delivery_reward(order, finish_t, self._t_limit)
        slack = order.et - finish_t
        return (
            1 if distance >= INF or reward <= 0.0 else 0,
            -reward / max(distance, 1),
            1 if finish_t > order.et else 0,
            max(slack, 0),
            distance,
            -order.p,
            order.id,
        )

    def _large_prefilter_key(self, shipper: Shipper, order: Order, t: int) -> Tuple[float, ...]:
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._manhattan(shipper.position, pickup)
        delivery_distance = self._manhattan(pickup, dropoff)
        route_distance = pickup_distance + delivery_distance
        finish_t = t + route_distance
        reward = delivery_reward(order, finish_t, self._t_limit)
        reward_per_step = reward / max(route_distance, 1)
        pressure = len(shipper.bag) / max(1, shipper.K_max)
        return (
            1 if reward <= 0.0 else 0,
            1 if finish_t > order.et else 0,
            -reward_per_step,
            pickup_distance,
            pressure,
            delivery_distance,
            order.et,
            -order.p,
            order.id,
        )

    def _large_pickup_rank(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> Tuple[float, ...]:
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._manhattan(shipper.position, pickup)
        delivery_distance = self._manhattan(pickup, dropoff)
        route_distance = pickup_distance + delivery_distance
        finish_t = t + route_distance
        reward = delivery_reward(order, finish_t, self._t_limit)
        reward_per_step = reward / max(route_distance, 1)
        reward_per_step *= 1.0 + 0.08 * self._hotspot_bonus(order)

        carried = self._carried_orders(shipper, orders)
        insertion_penalty = 0
        if carried:
            best_delivery = min(carried, key=lambda o: self._large_delivery_rank(shipper, o, t))
            direct = self._manhattan(shipper.position, (best_delivery.ex, best_delivery.ey))
            via_pickup = pickup_distance + self._manhattan(pickup, (best_delivery.ex, best_delivery.ey))
            insertion_penalty = max(0, via_pickup - direct)

        return (
            1 if reward <= 0.0 else 0,
            insertion_penalty,
            -reward_per_step,
            -0.02 * reward,
            1 if finish_t > order.et else 0,
            pickup_distance,
            delivery_distance,
            -order.p,
            order.et - finish_t,
            order.id,
        )

    def _assign_targets_large(self, obs: dict) -> Dict[int, Target]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        targets: Dict[int, Target] = {}

        for shipper in shippers:
            carried = self._carried_orders(shipper, orders)
            if not carried:
                continue
            order = min(carried, key=lambda o: self._large_delivery_rank(shipper, o, t))
            targets[shipper.id] = Target("deliver", order.id, (order.ex, order.ey))

        active_orders = [order for order in orders.values() if not order.picked and not order.delivered]
        if not active_orders:
            return targets

        candidate_limit = max(18, min(70, 3 * max(1, self._c)))
        exact_limit = max(3, min(5, self._c // 2))
        if self._n >= 70 or self._g >= 1000:
            candidate_limit = max(18, min(52, 2 * max(1, self._c)))
            exact_limit = 0

        pickup_pairs: List[Tuple[Tuple[float, ...], int, Order]] = []
        for shipper in shippers:
            if shipper.id in targets and len(shipper.bag) >= max(1, shipper.K_max):
                continue
            feasible = [order for order in active_orders if shipper.can_carry(order, orders)]
            if not feasible:
                continue
            if len(feasible) > candidate_limit:
                shortlisted = heapq.nsmallest(
                    candidate_limit,
                    feasible,
                    key=lambda o: self._large_prefilter_key(shipper, o, t),
                )
            else:
                shortlisted = sorted(feasible, key=lambda o: self._large_prefilter_key(shipper, o, t))
            ranked_orders = shortlisted[:exact_limit] if exact_limit > 0 else shortlisted
            for order in ranked_orders:
                rank = (
                    self._pickup_rank(shipper, order, orders, t)
                    if exact_limit > 0
                    else self._large_pickup_rank(shipper, order, orders, t)
                )
                if exact_limit > 0 and (rank[0] or rank[4] >= 0.0):
                    continue
                if exact_limit == 0 and rank[0] and rank[4]:
                    continue
                pickup_pairs.append((rank, shipper.id, order))

        used_shippers = set(targets)
        used_orders: set[int] = set()
        for _, shipper_id, order in sorted(pickup_pairs):
            if shipper_id in used_shippers or order.id in used_orders:
                continue
            targets[shipper_id] = Target("pickup", order.id, (order.sx, order.sy))
            used_shippers.add(shipper_id)
            used_orders.add(order.id)

        return targets

    def _assign_targets(self, obs: dict) -> Dict[int, Target]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        targets: Dict[int, Target] = {}

        for shipper in shippers:
            carried = self._carried_orders(shipper, orders)
            if not carried:
                continue
            order = min(carried, key=lambda o: self._delivery_rank(shipper, o, t))
            targets[shipper.id] = Target("deliver", order.id, (order.ex, order.ey))

        pickup_pairs: List[Tuple[Tuple[float, ...], int, Order]] = []
        for shipper in shippers:
            if shipper.id in targets and len(shipper.bag) >= max(1, shipper.K_max):
                continue
            for order in orders.values():
                if order.picked or order.delivered:
                    continue
                if not self._can_pickup(shipper, order, orders):
                    continue
                if self._expected_pickup_reward(shipper, order, t) <= 0.0:
                    continue
                pickup_pairs.append((self._beam_pickup_rank(shipper, order, orders, t), shipper.id, order))

        used_shippers = set(targets)
        used_orders: set[int] = set()
        for _, shipper_id, order in sorted(pickup_pairs):
            if shipper_id in used_shippers or order.id in used_orders:
                continue
            targets[shipper_id] = Target("pickup", order.id, (order.sx, order.sy))
            used_shippers.add(shipper_id)
            used_orders.add(order.id)

        return targets

    # ------------------------------------------------------------------
    # CBS
    # ------------------------------------------------------------------
    def _constraint_maps(
        self,
        constraints: Iterable[Constraint],
        shipper_id: int,
    ) -> Tuple[DefaultDict[int, set[Position]], DefaultDict[int, set[Tuple[Position, Position]]]]:
        vertex: DefaultDict[int, set[Position]] = defaultdict(set)
        edge: DefaultDict[int, set[Tuple[Position, Position]]] = defaultdict(set)
        for constraint in constraints:
            if constraint.shipper_id != shipper_id:
                continue
            if constraint.kind == "vertex":
                vertex[constraint.time].add(constraint.pos)
            elif constraint.kind == "edge" and constraint.to_pos is not None:
                edge[constraint.time].add((constraint.pos, constraint.to_pos))
        return vertex, edge

    def _violates(
        self,
        pos: Position,
        nxt: Position,
        depart_t: int,
        vertex_constraints: DefaultDict[int, set[Position]],
        edge_constraints: DefaultDict[int, set[Tuple[Position, Position]]],
    ) -> bool:
        return nxt in vertex_constraints[depart_t + 1] or (pos, nxt) in edge_constraints[depart_t]

    def _low_level_search(
        self,
        shipper: Shipper,
        target: Target,
        constraints: Tuple[Constraint, ...],
        max_time: int,
    ) -> Optional[List[Position]]:
        start = shipper.position
        vertex_constraints, edge_constraints = self._constraint_maps(constraints, shipper.id)
        if start in vertex_constraints[0]:
            return None

        open_heap: List[Tuple[int, int, int, Position, List[Position]]] = []
        start_h = self._distance(start, target.pos)
        if start_h >= INF:
            return None

        heapq.heappush(open_heap, (start_h, 0, 0, start, [start]))
        best_seen: Dict[Tuple[Position, int], int] = {(start, 0): 0}

        while open_heap:
            _, cost, t, pos, path = heapq.heappop(open_heap)
            if pos == target.pos:
                return path
            if t >= max_time:
                continue

            for _, nxt in self._neighbors(pos, include_wait=True):
                next_t = t + 1
                if self._violates(pos, nxt, t, vertex_constraints, edge_constraints):
                    continue
                h = self._distance(nxt, target.pos)
                if h >= INF or next_t + h > max_time:
                    continue
                state = (nxt, next_t)
                next_cost = cost + 1
                if best_seen.get(state, INF) <= next_cost:
                    continue
                best_seen[state] = next_cost
                heapq.heappush(open_heap, (next_cost + h, next_cost, next_t, nxt, path + [nxt]))

        return None

    def _path_pos(self, path: List[Position], t: int) -> Position:
        return path[t] if t < len(path) else path[-1]

    def _first_conflict(
        self,
        paths: Dict[int, List[Position]],
    ) -> Optional[Tuple[str, int, int, Position, Optional[Position], int]]:
        if len(paths) < 2:
            return None
        max_len = max(len(path) for path in paths.values())
        ids = sorted(paths)

        for t in range(max_len):
            occupied: Dict[Position, int] = {}
            for sid in ids:
                pos = self._path_pos(paths[sid], t)
                if pos in occupied:
                    return ("vertex", occupied[pos], sid, pos, None, t)
                occupied[pos] = sid

            if t + 1 >= max_len:
                continue
            for i, a in enumerate(ids):
                a_from = self._path_pos(paths[a], t)
                a_to = self._path_pos(paths[a], t + 1)
                for b in ids[i + 1 :]:
                    b_from = self._path_pos(paths[b], t)
                    b_to = self._path_pos(paths[b], t + 1)
                    if a_from == b_to and a_to == b_from and a_from != a_to:
                        return ("edge", a, b, a_from, a_to, t)
        return None

    def _push_cbs_node(
        self,
        heap: List[Tuple[int, int, int, Tuple[Constraint, ...], Dict[int, List[Position]]]],
        constraints: Tuple[Constraint, ...],
        paths: Dict[int, List[Position]],
    ) -> None:
        self._counter += 1
        cost = sum(len(path) - 1 for path in paths.values())
        conflicts_left = 1 if self._first_conflict(paths) else 0
        heapq.heappush(heap, (cost, conflicts_left, self._counter, constraints, paths))

    def _cbs_plan(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Target],
    ) -> Optional[Dict[int, List[Position]]]:
        active = [shipper for shipper in shippers if shipper.id in targets]
        idle = [shipper for shipper in shippers if shipper.id not in targets]
        if not active:
            return {}

        max_dist = max(self._distance(shipper.position, targets[shipper.id].pos) for shipper in active)
        if max_dist >= INF:
            return None
        max_time = min(self._planning_window, max(self._planning_window // 2, max_dist + 8))

        root_constraints: Tuple[Constraint, ...] = ()
        root_paths: Dict[int, List[Position]] = {}
        for shipper in active:
            path = self._low_level_search(shipper, targets[shipper.id], root_constraints, max_time)
            if path is None:
                return None
            root_paths[shipper.id] = path
        for shipper in idle:
            root_paths[shipper.id] = [shipper.position]

        open_heap: List[Tuple[int, int, int, Tuple[Constraint, ...], Dict[int, List[Position]]]] = []
        self._push_cbs_node(open_heap, root_constraints, root_paths)
        expanded = 0

        while open_heap and expanded < self._max_cbs_nodes:
            _, _, _, constraints, paths = heapq.heappop(open_heap)
            expanded += 1
            conflict = self._first_conflict(paths)
            if conflict is None:
                return paths

            kind, sid_a, sid_b, pos, to_pos, conflict_t = conflict
            constrained_sids = [sid for sid in (sid_a, sid_b) if sid in targets]
            if not constrained_sids:
                return paths

            for sid in constrained_sids:
                if kind == "vertex":
                    new_constraint = Constraint(sid, "vertex", pos, conflict_t)
                else:
                    assert to_pos is not None
                    if sid == sid_a:
                        new_constraint = Constraint(sid, "edge", pos, conflict_t, to_pos)
                    else:
                        new_constraint = Constraint(sid, "edge", to_pos, conflict_t, pos)

                new_constraints = constraints + (new_constraint,)
                new_paths = dict(paths)
                shipper = next(s for s in active if s.id == sid)
                new_path = self._low_level_search(shipper, targets[sid], new_constraints, max_time)
                if new_path is None:
                    continue
                new_paths[sid] = new_path
                self._push_cbs_node(open_heap, new_constraints, new_paths)

        return None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def _action_from_target_path(
        self,
        shipper: Shipper,
        target: Target,
        path: Optional[List[Position]],
    ) -> Action:
        nxt = path[1] if path is not None and len(path) >= 2 else shipper.position
        move = self._move_between(shipper.position, nxt)
        arrival = valid_next_pos(shipper.position, move, self.grid)
        if arrival != target.pos:
            return move, 0
        if target.kind == "pickup":
            return move, 1
        if target.kind == "deliver":
            return move, 2
        return move, 0

    def _fallback_actions(self, obs: dict, targets: Dict[int, Target]) -> Dict[int, Action]:
        actions: Dict[int, Action] = {}
        reserved: set[Position] = set()
        shippers: List[Shipper] = obs["shippers"]

        for shipper in sorted(shippers, key=lambda s: (0 if s.id in targets else 1, s.id)):
            target = targets.get(shipper.id)
            if target is None:
                actions[shipper.id] = ("S", 0)
                reserved.add(shipper.position)
                continue

            best: Tuple[int, Move, Position] = (INF, "S", shipper.position)
            for move, nxt in self._neighbors(shipper.position, include_wait=True):
                if nxt in reserved:
                    continue
                score = self._distance(nxt, target.pos)
                if score < best[0]:
                    best = (score, move, nxt)
            _, move, nxt = best
            op = 0
            if nxt == target.pos:
                op = 1 if target.kind == "pickup" else 2
            actions[shipper.id] = (move, op)
            reserved.add(nxt)

        return actions

    def _large_action_from_target(
        self,
        shipper: Shipper,
        target: Target,
        reserved: set[Position],
    ) -> Tuple[Action, Position]:
        move = self._next_move(shipper.position, target.pos)
        nxt = valid_next_pos(shipper.position, move, self.grid)

        if nxt in reserved:
            choices: List[Tuple[int, int, Move, Position]] = []
            for alt_move, alt_nxt in self._neighbors(shipper.position, include_wait=True):
                if alt_nxt in reserved:
                    continue
                choices.append((
                    self._manhattan(alt_nxt, target.pos),
                    0 if alt_move == move else 1,
                    alt_move,
                    alt_nxt,
                ))
            if choices:
                _, _, move, nxt = min(choices)
            else:
                move, nxt = "S", shipper.position

        op = 0
        if nxt == target.pos:
            op = 1 if target.kind == "pickup" else 2
        return (move, op), nxt

    def _decide_actions_large(self, obs: dict) -> Dict[int, Action]:
        targets = self._assign_targets_large(obs)
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))

        def priority(shipper: Shipper) -> Tuple[float, ...]:
            target = targets.get(shipper.id)
            if target is None:
                return (2, shipper.id)
            if target.kind == "deliver":
                order = orders.get(target.order_id)
                if order is None:
                    return (1, shipper.id)
                return (0, order.et - t, -order.p, shipper.id)
            order = orders.get(target.order_id)
            if order is None:
                return (1, shipper.id)
            return (1, order.et - t, self._manhattan(shipper.position, target.pos), -order.p, shipper.id)

        actions: Dict[int, Action] = {}
        reserved: set[Position] = set()
        for shipper in sorted(shippers, key=priority):
            target = targets.get(shipper.id)
            if target is None:
                if shipper.position not in reserved:
                    reserved.add(shipper.position)
                actions[shipper.id] = ("S", 0)
                continue
            action, nxt = self._large_action_from_target(shipper, target, reserved)
            actions[shipper.id] = action
            reserved.add(nxt)
        return actions

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        self._refresh_from_obs(obs)
        self._hotspot_tracker.update(obs)
        if self._large_mode:
            return self._decide_actions_large(obs)

        targets = self._assign_targets(obs)
        shippers: List[Shipper] = obs["shippers"]
        paths = self._cbs_plan(shippers, targets)
        if paths is None:
            return self._fallback_actions(obs, targets)

        actions: Dict[int, Action] = {}
        for shipper in shippers:
            target = targets.get(shipper.id)
            if target is None:
                actions[shipper.id] = ("S", 0)
                continue
            actions[shipper.id] = self._action_from_target_path(shipper, target, paths.get(shipper.id))
        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        self._hotspot_tracker.reset()
        self._refresh_from_obs(obs)

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
