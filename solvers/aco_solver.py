from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, Any]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class ACOSolver(Solver):
    method_name = "ACO_Strict_Online"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

        self._path_cache: Dict[Tuple[Position, Position], Tuple[int, Move]] = {}
        self._pheromone: Dict[Tuple[str, int], float] = {}
        self._dynamic_hotspots: Dict[Position, float] = {}

        self._alpha = 1.0
        self._beta = 2.2
        self._rho = 0.05
        self._min_pheromone = 0.2
        self._max_pheromone = 15.0

        self._N = len(self.grid)
        self._path_cache_limit = 50_000 if self._N >= 50 else 200_000

    def _candidate_limit(self) -> int:
        if self._N >= 100:
            return 11
        if self._N >= 80:
            return 15
        if self._N >= 50:
            return 20
        if self._N >= 30:
            return 40
        return 100

    def _enable_extra_pickup(self) -> bool:
        return 12 <= self._N <= 30

    def _manhattan(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {
            start: (None, "S")
        }

        while queue:
            cur = queue.popleft()

            if cur == goal:
                return parent

            for move, nxt in self._neighbors(cur):
                if nxt in parent:
                    continue
                parent[nxt] = (cur, move)
                queue.append(nxt)

        return None

    def _compute_path_properties(self, start: Position, goal: Position) -> Tuple[int, Move]:
        if start == goal:
            return 0, "S"

        key = (start, goal)

        if key in self._path_cache:
            return self._path_cache[key]

        if len(self._path_cache) > self._path_cache_limit:
            self._path_cache.clear()

        parent = self._bfs_parents(start, goal)

        if parent is None or goal not in parent:
            self._path_cache[key] = (INF, "S")
            return INF, "S"

        dist = 0
        cur = goal
        first_move = "S"

        while cur != start:
            prev, move = parent[cur]

            if prev is None:
                self._path_cache[key] = (INF, "S")
                return INF, "S"

            if prev == start:
                first_move = move

            cur = prev
            dist += 1

        self._path_cache[key] = (dist, first_move)
        return dist, first_move

    def _distance(self, start: Position, goal: Position) -> int:
        return self._compute_path_properties(start, goal)[0]

    def _next_move(self, start: Position, goal: Position) -> Move:
        return self._compute_path_properties(start, goal)[1]

    def _step_travel_cost(self, move: Move, current_w: float, max_w: float) -> float:
        if move == "S":
            return 0.0
        return -0.01 * (1.0 + current_w / max(max_w, 1.0))

    def _current_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _can_carry_online(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
    ) -> bool:
        if order.picked or order.delivered:
            return False

        if len(shipper.bag) >= shipper.K_max:
            return False

        current_w = self._current_weight(shipper, orders)
        return current_w + order.w <= shipper.W_max

    def _update_hotspot_beliefs(self, orders: Dict[int, Order]) -> None:
        self._dynamic_hotspots.clear()

        for order in orders.values():
            if not order.picked and not order.delivered:
                pos = (order.sx, order.sy)
                self._dynamic_hotspots[pos] = self._dynamic_hotspots.get(pos, 0.0) + 1.0

    def _get_hotspot_bonus(self, pos: Position) -> float:
        bonus = 0.0

        for h_pos, weight in self._dynamic_hotspots.items():
            dist = self._manhattan(pos, h_pos)
            if dist <= 3:
                bonus += weight / (1.0 + dist)

        return min(3.0, bonus)

    def _pheromone_key(self, task_type: str, order_id: int) -> Tuple[str, int]:
        return task_type, order_id

    def _get_pheromone(self, task_type: str, order_id: int) -> float:
        return self._pheromone.get(self._pheromone_key(task_type, order_id), 1.0)

    def _evaporate_pheromone(self, active_order_ids: Set[int]) -> None:
        for key in list(self._pheromone.keys()):
            _, oid = key

            if oid not in active_order_ids:
                self._pheromone.pop(key, None)
                continue

            value = max(self._min_pheromone, self._pheromone[key] * (1.0 - self._rho))

            if math.isclose(value, self._min_pheromone):
                self._pheromone.pop(key, None)
            else:
                self._pheromone[key] = value

    def _reinforce(self, task_type: str, order_id: int, amount: float) -> None:
        key = self._pheromone_key(task_type, order_id)
        current = self._pheromone.get(key, 1.0)
        self._pheromone[key] = min(self._max_pheromone, current + amount)

    def _unpicked_candidates_for_shipper(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        limit: Optional[int] = None,
    ) -> List[Order]:
        limit = self._candidate_limit() if limit is None else limit

        candidates = [
            o for o in orders.values()
            if not o.picked
            and not o.delivered
            and self._can_carry_online(shipper, o, orders)
        ]

        if self._N >= 50:
            max_pickup_dist = max(18, self._N // 2)
            filtered: List[Order] = []

            for o in candidates:
                pickup_pos = (o.sx, o.sy)
                delivery_pos = (o.ex, o.ey)

                d_pick = self._manhattan(shipper.position, pickup_pos)
                d_del = self._manhattan(pickup_pos, delivery_pos)

                if d_pick > max_pickup_dist:
                    continue

                est_finish = d_pick + d_del
                slack = o.et - est_finish

                if self._N >= 80 and slack < -120 and o.p <= 2:
                    continue

                filtered.append(o)

            candidates = filtered

        candidates.sort(
            key=lambda o: (
                -o.p,
                self._manhattan(shipper.position, (o.sx, o.sy))
                + 0.35 * self._manhattan((o.sx, o.sy), (o.ex, o.ey)),
                o.et,
                o.id,
            )
        )

        return candidates[:limit]

    def _best_pickup_at_pos(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        pos: Position,
    ) -> Optional[Order]:
        candidates = [
            o for o in orders.values()
            if not o.picked
            and not o.delivered
            and (o.sx, o.sy) == pos
            and self._can_carry_online(shipper, o, orders)
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda o: (o.p, -o.et, -o.id))

    def _pickup_heuristic(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        T_max: int,
    ) -> float:
        pickup_pos = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)

        d1 = self._distance(shipper.position, pickup_pos)
        if d1 >= INF:
            return 0.0

        d2 = self._distance(pickup_pos, delivery_pos)
        if d2 >= INF:
            return 0.0

        finish_t = t + d1 + d2
        est_reward = delivery_reward(order, finish_t, T_max)

        current_w = self._current_weight(shipper, orders)

        cost_d1 = abs(self._step_travel_cost("U", current_w, shipper.W_max)) * d1
        cost_d2 = abs(self._step_travel_cost("U", current_w + order.w, shipper.W_max)) * d2

        hotspot_bonus = self._get_hotspot_bonus(pickup_pos)

        remaining = order.et - finish_t
        urgency_bonus = 1.0 / (1.0 + remaining) if remaining >= 0 else -0.5

        net_profit = est_reward - cost_d1 - cost_d2 + hotspot_bonus + urgency_bonus

        if finish_t <= order.et:
            net_profit *= 1.25
        elif self._N >= 80 and order.p <= 2:
            net_profit *= 0.75

        if self._N >= 50:
            return max(0.01, net_profit) / (1.0 + 1.5 * d1 + 0.4 * d2)

        return max(0.01, net_profit) / (1.0 + d1 + 0.5 * d2)

    def _best_delivery_target(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        T_max: int,
    ) -> Optional[Order]:
        bag_orders = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

        if not bag_orders:
            return None

        def score(o: Order):
            d = self._distance(shipper.position, (o.ex, o.ey))

            if d >= INF:
                return (-1e18, -1e18, -1e18, -1e18, -1e18)

            arrive_t = t + d
            reward = delivery_reward(o, arrive_t, T_max)
            lateness = max(0, arrive_t - o.et)

            return (
                reward / (1.0 + d),
                -lateness,
                o.p,
                -d,
                -o.id,
            )

        return max(bag_orders, key=score)

    def _assign_pickups(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        t: int,
        T_max: int,
    ) -> Dict[int, Order]:
        assignments: Dict[int, Order] = {}

        if not shippers or not orders:
            return assignments

        available_shippers = set(s.id for s in shippers)
        shipper_map = {s.id: s for s in shippers}
        taken_orders: Set[int] = set()

        while available_shippers:
            best_candidate: Optional[Tuple[int, Order]] = None
            best_rank = (INF, INF, INF, INF)

            for sid in list(available_shippers):
                shipper = shipper_map[sid]

                for order in self._unpicked_candidates_for_shipper(shipper, orders):
                    if order.id in taken_orders:
                        continue

                    h_val = self._pickup_heuristic(shipper, order, orders, t, T_max)

                    if h_val <= 0.0:
                        continue

                    score = (
                        self._get_pheromone("pickup", order.id) ** self._alpha
                    ) * (h_val ** self._beta)

                    manh = self._manhattan(shipper.position, (order.sx, order.sy))

                    if self._N >= 50:
                        rank = (
                            -score,
                            manh,
                            -order.p,
                            order.et,
                        )
                    else:
                        rank = (
                            -order.p,
                            -score,
                            manh,
                            order.id,
                        )

                    if rank < best_rank:
                        best_rank = rank
                        best_candidate = (sid, order)

            if best_candidate is None:
                break

            sid, order = best_candidate
            assignments[sid] = order
            taken_orders.add(order.id)
            available_shippers.remove(sid)

        return assignments

    def _has_delivery_after_move(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        next_pos: Position,
    ) -> bool:
        for oid in shipper.bag:
            if oid not in orders:
                continue

            order = orders[oid]

            if order.delivered:
                continue

            if (order.ex, order.ey) == next_pos:
                return True

        return False

    def _find_alternative_move(
        self,
        current: Position,
        blocked_positions: Set[Position],
    ) -> Move:
        for move in MOVES:
            nxt = valid_next_pos(current, move, self.grid)

            if nxt != current and nxt not in blocked_positions:
                return move

        return "S"

    def _maybe_extra_pickup_before_delivery(
        self,
        shipper: Shipper,
        delivery_order: Optional[Order],
        orders: Dict[int, Order],
    ) -> Optional[Order]:
        if delivery_order is None:
            return None

        if not self._enable_extra_pickup():
            return None

        if len(shipper.bag) >= shipper.K_max:
            return None

        delivery_pos = (delivery_order.ex, delivery_order.ey)
        delivery_dist = self._distance(shipper.position, delivery_pos)

        if delivery_dist >= INF:
            return None

        best_extra: Optional[Order] = None
        best_rank = (INF, INF, INF, INF)

        for order in self._unpicked_candidates_for_shipper(shipper, orders, limit=20):
            pickup_pos = (order.sx, order.sy)
            pickup_dist = self._distance(shipper.position, pickup_pos)

            if pickup_dist >= INF:
                continue

            if pickup_dist <= 3 and pickup_dist + 2 < delivery_dist:
                rank = (
                    pickup_dist,
                    -order.p,
                    order.et,
                    order.id,
                )

                if rank < best_rank:
                    best_rank = rank
                    best_extra = order

        return best_extra

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))
        T_max = int(obs.get("T", 240))

        self.grid = obs["grid"]
        self._N = int(obs.get("N", len(self.grid)))
        self._path_cache_limit = 50_000 if self._N >= 50 else 200_000

        self._update_hotspot_beliefs(orders)

        actions: Dict[int, Action] = {}
        occupied_next_positions: Set[Position] = set()

        delivery_targets: Dict[int, Optional[Order]] = {}
        idle_shippers: List[Shipper] = []

        for shipper in shippers:
            target = self._best_delivery_target(shipper, orders, t, T_max)
            delivery_targets[shipper.id] = target

            if target is None:
                idle_shippers.append(shipper)

        assigned_pickups = self._assign_pickups(idle_shippers, orders, t, T_max)

        for shipper in sorted(shippers, key=lambda s: s.id):
            move: Move = "S"
            cargo_op: Any = 0

            delivery_order = delivery_targets.get(shipper.id)
            pickup_order = assigned_pickups.get(shipper.id)

            extra_pickup = self._maybe_extra_pickup_before_delivery(
                shipper,
                delivery_order,
                orders,
            )

            if extra_pickup is not None:
                pickup_order = extra_pickup
                delivery_order = None

            if delivery_order is not None:
                target_pos = (delivery_order.ex, delivery_order.ey)
                move = self._next_move(shipper.position, target_pos)

                if move not in MOVES:
                    move = "S"

                next_pos = valid_next_pos(shipper.position, move, self.grid)

                if move != "S" and next_pos in occupied_next_positions:
                    move = self._find_alternative_move(shipper.position, occupied_next_positions)
                    next_pos = valid_next_pos(shipper.position, move, self.grid)

                if self._has_delivery_after_move(shipper, orders, next_pos):
                    cargo_op = 2
                    self._reinforce(
                        "deliver",
                        delivery_order.id,
                        delivery_reward(delivery_order, t + 1, T_max) * 0.05,
                    )

            elif pickup_order is not None:
                target_pos = (pickup_order.sx, pickup_order.sy)
                move = self._next_move(shipper.position, target_pos)

                if move not in MOVES:
                    move = "S"

                next_pos = valid_next_pos(shipper.position, move, self.grid)

                if move != "S" and next_pos in occupied_next_positions:
                    move = self._find_alternative_move(shipper.position, occupied_next_positions)
                    next_pos = valid_next_pos(shipper.position, move, self.grid)

                best_here = self._best_pickup_at_pos(shipper, orders, next_pos)

                if best_here is not None:
                    cargo_op = 1
                    pickup_order = best_here

                    finish_t = t + 1 + self._distance(
                        next_pos,
                        (pickup_order.ex, pickup_order.ey),
                    )

                    self._reinforce(
                        "pickup",
                        pickup_order.id,
                        delivery_reward(pickup_order, finish_t, T_max) * 0.05,
                    )

            else:
                move = "S"
                next_pos = shipper.position

                if self._has_delivery_after_move(shipper, orders, next_pos):
                    cargo_op = 2

                elif self._best_pickup_at_pos(shipper, orders, next_pos) is not None:
                    cargo_op = 1

            occupied_next_positions.add(valid_next_pos(shipper.position, move, self.grid))
            actions[shipper.id] = (move, cargo_op)

        return actions

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