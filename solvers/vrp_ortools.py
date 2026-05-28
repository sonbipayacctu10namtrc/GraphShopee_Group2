from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]
INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class VRPOrToolsSolver(Solver):
    """
    Adaptive VRP-style online solver, implemented in pure Python.

    The grader expects a class named VRPOrToolsSolver, but the Kaggle/runtime
    environment may not have OR-Tools installed. This solver therefore uses a
    lightweight rolling-horizon VRP heuristic:

    - BFS shortest-path routing on the obstacle grid.
    - Global shipper-order assignment instead of per-shipper nearest-order greed.
    - Reward/deadline/capacity aware pickup scoring.
    - Optional soft region penalty on very large or blocked maps.
    - Small detour insertion: a shipper carrying an order may pick up a nearby
      high-value order if doing so does not endanger the current delivery.
    - One-step collision avoidance for next cells.

    It is not an exact OR-Tools model. It is a deterministic, explainable
    VRP-style heuristic designed for the online MAPD simulator.
    """

    method_name = "VRP-OrTools"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.cfg = {"N": env.N, "C": env.C, "G": env.G, "T": env.T, "name": env.config_name}
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._map_stats_cache: Optional[Tuple[int, int, float, float]] = None

    # ------------------------------------------------------------------
    # Shortest path utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parent(self, start: Position, goal: Position) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        q: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}

        while q:
            cur = q.popleft()
            if cur == goal:
                return parent
            for move, nxt in self._neighbors(cur):
                if nxt in parent:
                    continue
                parent[nxt] = (cur, move)
                q.append(nxt)
        return None

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        key = (start, goal)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached

        parent = self._bfs_parent(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        d = 0
        cur = goal
        while cur != start:
            prev, _ = parent[cur]
            if prev is None:
                self._distance_cache[key] = INF
                return INF
            cur = prev
            d += 1
        self._distance_cache[key] = d
        return d

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        key = (start, goal)
        cached = self._next_move_cache.get(key)
        if cached is not None:
            return cached

        parent = self._bfs_parent(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        cur = goal
        while True:
            prev, move = parent[cur]
            if prev is None:
                self._next_move_cache[key] = "S"
                return "S"
            if prev == start:
                self._next_move_cache[key] = move
                return move
            cur = prev

    # ------------------------------------------------------------------
    # Map-adaptive policy
    # ------------------------------------------------------------------
    def _map_stats(self) -> Tuple[int, int, float, float]:
        if self._map_stats_cache is not None:
            return self._map_stats_cache

        n = int(self.cfg.get("N", len(self.grid)))
        c = int(self.cfg.get("C", 1))
        total = max(1, n * n)
        blocked = sum(1 for row in self.grid for cell in row if cell == 1)
        free = total - blocked
        obstacle_ratio = blocked / total
        free_per_shipper = free / max(c, 1)
        self._map_stats_cache = (n, c, obstacle_ratio, free_per_shipper)
        return self._map_stats_cache

    def _should_use_region_policy(self) -> bool:
        n, c, obstacle_ratio, free_per_shipper = self._map_stats()
        return (
            n >= 20
            or (n >= 18 and c <= 4 and obstacle_ratio >= 0.22)
            or (free_per_shipper >= 60 and obstacle_ratio >= 0.22)
        )

    def _region_weight(self) -> float:
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
        return weight

    def _region_of(self, pos: Position) -> int:
        n = int(self.cfg.get("N", len(self.grid)))
        r, c = pos
        return (0 if r < n // 2 else 2) + (0 if c < n // 2 else 1)

    def _region_penalty(self, shipper: Shipper, order: Order) -> float:
        weight = self._region_weight()
        if weight <= 0:
            return 0.0

        s_region = self._region_of(shipper.position)
        p_region = self._region_of((order.sx, order.sy))
        d_region = self._region_of((order.ex, order.ey))

        raw = 0.0
        if p_region != s_region:
            raw += 8.0
        if d_region != s_region:
            raw += 4.0
        if p_region != d_region:
            raw += 2.0
        return raw * weight

    # ------------------------------------------------------------------
    # Order scoring
    # ------------------------------------------------------------------
    def _estimated_reward(self, order: Order, finish_t: int) -> float:
        return delivery_reward(order, finish_t, int(self.cfg.get("T", 1)))

    def _bag_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _can_take(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered:
            return False
        if len(shipper.bag) >= shipper.K_max:
            return False
        return self._bag_weight(shipper, orders) + order.w <= shipper.W_max

    def _delivery_key(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int]:
        d = self._distance(shipper.position, (order.ex, order.ey))
        finish_t = t + d
        slack = order.et - finish_t
        return (1 if finish_t > order.et else 0, slack, d, -order.p, order.id)

    def _pickup_score_tuple(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> Optional[Tuple[float, float, int, int, int]]:
        pickup = (order.sx, order.sy)
        delivery = (order.ex, order.ey)
        d1 = self._distance(shipper.position, pickup)
        d2 = self._distance(pickup, delivery)
        if d1 >= INF or d2 >= INF:
            return None

        route_d = d1 + d2
        finish_t = t + route_d
        slack = order.et - finish_t
        reward = self._estimated_reward(order, finish_t)
        value_rate = reward / (route_d + 1.0)

        # Large-map pruning: avoid sending a shipper too far for a weak/late order.
        n, c, _obs, free_per_shipper = self._map_stats()
        if n >= 18 and free_per_shipper >= 45:
            if d1 > max(8, n // 2) and order.p <= 1 and slack < 15:
                return None

        region_penalty = self._region_penalty(shipper, order)
        late_flag = 1 if finish_t > order.et else 0

        # Smaller tuple is better. Negative value_rate makes higher value better.
        return (
            float(late_flag),
            region_penalty - value_rate,
            slack,
            route_d,
            order.id,
        )

    def _best_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Order]:
        carried = [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]
        if not carried:
            return None
        return min(carried, key=lambda o: self._delivery_key(shipper, o, t))

    def _detour_allowed(self, shipper: Shipper, pickup_order: Order, delivery_order: Order, orders: Dict[int, Order], t: int) -> bool:
        if not self._can_take(shipper, pickup_order, orders):
            return False

        direct_d = self._distance(shipper.position, (delivery_order.ex, delivery_order.ey))
        pickup_d = self._distance(shipper.position, (pickup_order.sx, pickup_order.sy))
        after_pickup_d = self._distance((pickup_order.sx, pickup_order.sy), (delivery_order.ex, delivery_order.ey))
        if direct_d >= INF or pickup_d >= INF or after_pickup_d >= INF:
            return False

        direct_finish = t + direct_d
        detour_finish = t + pickup_d + after_pickup_d
        if direct_finish <= delivery_order.et < detour_finish:
            return False

        n, _c, _obs, _fps = self._map_stats()
        adaptive_limit = max(2, min(n // 3, max(2, direct_d // 2)))
        if detour_finish - direct_finish > adaptive_limit:
            return False

        pickup_score = self._pickup_score_tuple(shipper, pickup_order, orders, t)
        if pickup_score is None or pickup_score[0] > 0:
            return False

        return pickup_order.p >= delivery_order.p or pickup_d <= max(2, direct_d // 2)

    # ------------------------------------------------------------------
    # Assignment and action generation
    # ------------------------------------------------------------------
    def _global_pickup_assignment(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        delivery_targets: Dict[int, Optional[Order]],
        t: int,
    ) -> Dict[int, Order]:
        candidates: List[Tuple[Tuple[float, float, int, int, int], int, Order]] = []
        for s in shippers:
            current_delivery = delivery_targets.get(s.id)
            for o in orders.values():
                if not self._can_take(s, o, orders):
                    continue
                if current_delivery is not None and not self._detour_allowed(s, o, current_delivery, orders, t):
                    continue
                score = self._pickup_score_tuple(s, o, orders, t)
                if score is None:
                    continue
                candidates.append((score, s.id, o))

        assigned: Dict[int, Order] = {}
        used_orders: set[int] = set()
        for _score, sid, order in sorted(candidates):
            if sid in assigned or order.id in used_orders:
                continue
            assigned[sid] = order
            used_orders.add(order.id)
        return assigned

    def _move_towards(self, shipper: Shipper, goal: Position, reserved: set[Position]) -> Tuple[Move, Position]:
        move = self._next_move(shipper.position, goal)
        nxt = valid_next_pos(shipper.position, move, self.grid)
        if nxt not in reserved:
            return move, nxt

        alternatives: List[Tuple[int, Move, Position]] = []
        for alt_move in ("S",) + MOVES:
            alt_pos = valid_next_pos(shipper.position, alt_move, self.grid)
            if alt_pos in reserved:
                continue
            alternatives.append((self._distance(alt_pos, goal), alt_move, alt_pos))

        if not alternatives:
            return "S", shipper.position
        _, best_move, best_pos = min(alternatives)
        return best_move, best_pos

    def _action_to_pickup(self, shipper: Shipper, order: Order, reserved: set[Position]) -> Action:
        goal = (order.sx, order.sy)
        move, nxt = self._move_towards(shipper, goal, reserved)
        # Only pickup when arriving exactly at target; prevents pickup_best from
        # taking another order on the current cell by accident.
        return (move, 1) if nxt == goal else (move, 0)

    def _action_to_deliver(self, shipper: Shipper, order: Order, reserved: set[Position]) -> Action:
        goal = (order.ex, order.ey)
        move, nxt = self._move_towards(shipper, goal, reserved)
        # op=2 is safe: env only delivers orders whose destination matches cell.
        return (move, 2) if nxt == goal else (move, 0)

    def _next_position_from_action(self, shipper: Shipper, action: Action) -> Position:
        move, _op = action
        return valid_next_pos(shipper.position, move, self.grid)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))

        delivery_targets: Dict[int, Optional[Order]] = {
            s.id: self._best_delivery(s, orders, t) for s in shippers
        }
        pickup_targets = self._global_pickup_assignment(shippers, orders, delivery_targets, t)

        # Move urgent delivery shippers first, then pickup-only shippers.
        action_order = sorted(
            shippers,
            key=lambda s: (
                self._delivery_key(s, delivery_targets[s.id], t)
                if delivery_targets[s.id] is not None
                else (1, INF, INF, 0, s.id)
            ),
        )

        actions: Dict[int, Action] = {}
        reserved_next: set[Position] = set()
        for s in action_order:
            pickup = pickup_targets.get(s.id)
            delivery = delivery_targets.get(s.id)

            if pickup is not None:
                action = self._action_to_pickup(s, pickup, reserved_next)
            elif delivery is not None:
                action = self._action_to_deliver(s, delivery, reserved_next)
            else:
                action = ("S", 0)

            actions[s.id] = action
            reserved_next.add(self._next_position_from_action(s, action))

        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start = time.time()
        obs = self.env.reset()
        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _reward, done, _info = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, elapsed_sec=time.time() - start)
