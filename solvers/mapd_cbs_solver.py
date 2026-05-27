from __future__ import annotations

import time
import heapq
from collections import Counter, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class MAPDCBSSolver(Solver):
    """Online MAPD solver with a lightweight CBS-style conflict filter."""

    method_name = "MAPD-CBS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._path_cache: Dict[Position, Tuple[Dict[Position, int], Dict[Position, Move]]] = {}
        self._recent_pickups: deque[Position] = deque(maxlen=20)
        self._map_size = 0
        self._shipper_count = 0

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _paths_from(self, start: Position) -> Tuple[Dict[Position, int], Dict[Position, Move]]:
        cached = self._path_cache.get(start)
        if cached is not None:
            return cached

        if not is_valid_cell(start, self.grid):
            empty: Tuple[Dict[Position, int], Dict[Position, Move]] = ({}, {})
            self._path_cache[start] = empty
            return empty

        distances: Dict[Position, int] = {start: 0}
        first_moves: Dict[Position, Move] = {start: "S"}
        queue: deque[Position] = deque([start])

        while queue:
            current = queue.popleft()
            for move, nxt in self._neighbors(current):
                if nxt in distances:
                    continue
                distances[nxt] = distances[current] + 1
                first_moves[nxt] = move if current == start else first_moves[current]
                queue.append(nxt)

        cached = (distances, first_moves)
        self._path_cache[start] = cached
        return cached

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0

        distances, _ = self._paths_from(start)
        return distances.get(goal, INF)

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"

        _, first_moves = self._paths_from(start)
        return first_moves.get(goal, "S")

    # ------------------------------------------------------------------
    # Task scoring
    # ------------------------------------------------------------------
    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _can_carry(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered:
            return False
        if len(shipper.bag) >= shipper.K_max:
            return False
        return self._carried_weight(shipper, orders) + order.w <= shipper.W_max

    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

    def _carried_delivery_cost(self, shipper: Shipper, orders: Dict[int, Order]) -> int:
        carried = self._carried_orders(shipper, orders)
        if not carried:
            return 0
        return min(self._distance(shipper.position, (order.ex, order.ey)) for order in carried)

    def _delivery_key(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int]:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        finish_t = t + distance
        if self._map_size < 18:
            return (
                1 if finish_t > order.et else 0,
                -order.p,
                order.et - finish_t,
                distance,
                order.id,
            )
        return (
            1 if finish_t > order.et else 0,
            order.et - finish_t,
            distance,
            -order.p,
            order.id,
        )

    def _pickup_key(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> Tuple[float, ...]:
        pickup = (order.sx, order.sy)
        delivery = (order.ex, order.ey)
        pickup_dist = self._distance(shipper.position, pickup)
        delivery_dist = self._distance(pickup, delivery)
        finish_t = t + pickup_dist + delivery_dist

        insertion_penalty = 0
        map_size = self._map_size
        if 12 <= map_size < 18:
            nearest_delivery = self._carried_delivery_cost(shipper, orders)
            delivery_order = self._select_delivery(shipper, orders, t)
            if nearest_delivery > 0 and delivery_order is not None:
                insertion_penalty = (
                    pickup_dist
                    + self._distance(pickup, (delivery_order.ex, delivery_order.ey))
                    - nearest_delivery
                )

        return (
            1 if finish_t > order.et else 0,
            insertion_penalty,
            pickup_dist + 0.5 * delivery_dist
            if map_size >= 19
            else pickup_dist + 0.25 * delivery_dist
            if 13 <= map_size < 15 and self._shipper_count >= 4
            else pickup_dist,
            pickup_dist,
            -order.p,
            order.et - finish_t,
            delivery_dist,
            order.id,
        )

    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Order]:
        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried:
            return None
        return min(carried, key=lambda order: self._delivery_key(shipper, order, t))

    def _rough_pickup_key(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int, int]:
        pickup_dist = abs(shipper.r - order.sx) + abs(shipper.c - order.sy)
        delivery_dist = abs(order.sx - order.ex) + abs(order.sy - order.ey)
        finish_t = t + pickup_dist + delivery_dist
        return (
            1 if finish_t > order.et else 0,
            order.et - finish_t,
            pickup_dist + delivery_dist,
            pickup_dist,
            -order.p,
            order.id,
        )

    def _pickup_candidate_limit(self, visible_count: int) -> int:
        if visible_count <= 160:
            return visible_count
        return min(220, max(80, 60 + 8 * self._shipper_count))

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        t: int,
    ) -> Optional[Order]:
        if len(shipper.bag) >= shipper.K_max:
            return None

        carried_weight = self._carried_weight(shipper, orders)
        rough_candidates: List[Order] = []
        for order in orders.values():
            if order.id in reserved_order_ids or order.picked or order.delivered:
                continue
            if carried_weight + order.w > shipper.W_max:
                continue
            rough_candidates.append(order)

        limit = self._pickup_candidate_limit(len(rough_candidates))
        if len(rough_candidates) > limit:
            rough_candidates = heapq.nsmallest(
                limit,
                rough_candidates,
                key=lambda order: self._rough_pickup_key(shipper, order, t),
            )

        candidates: List[Order] = []

        for order in rough_candidates:
            pickup_dist = self._distance(shipper.position, (order.sx, order.sy))
            delivery_dist = self._distance((order.sx, order.sy), (order.ex, order.ey))
            if pickup_dist >= INF or delivery_dist >= INF:
                continue
            candidates.append(order)

        if not candidates:
            return None
        return min(candidates, key=lambda order: self._pickup_key(shipper, order, orders, t))

    def _should_pickup_before_delivery(
        self,
        shipper: Shipper,
        pickup_order: Order,
        delivery_order: Order,
        orders: Dict[int, Order],
        t: int,
    ) -> bool:
        map_size = self._map_size
        if map_size < 12:
            return False
        if len(shipper.bag) >= shipper.K_max:
            return False

        if map_size == 12:
            pickup_pos = (pickup_order.sx, pickup_order.sy)
            delivery_pos = (delivery_order.ex, delivery_order.ey)
            direct_finish = t + self._distance(shipper.position, delivery_pos)
            inserted_finish = (
                t
                + self._distance(shipper.position, pickup_pos)
                + self._distance(pickup_pos, delivery_pos)
            )
            if inserted_finish >= INF:
                return False
            if direct_finish <= delivery_order.et < inserted_finish:
                return False
            return inserted_finish - direct_finish <= 3 and pickup_order.p >= delivery_order.p

        if map_size >= 18:
            direct = self._distance(shipper.position, (delivery_order.ex, delivery_order.ey))
            via_pickup = self._distance(shipper.position, (pickup_order.sx, pickup_order.sy))
            pickup_to_delivery = self._distance(
                (pickup_order.sx, pickup_order.sy),
                (delivery_order.ex, delivery_order.ey),
            )
            if direct >= INF or via_pickup >= INF or pickup_to_delivery >= INF:
                return False

            direct_finish = t + direct
            detour_finish = t + via_pickup + pickup_to_delivery
            if direct_finish <= delivery_order.et < detour_finish:
                return False

            detour_extra = detour_finish - direct_finish
            if map_size >= 19 and via_pickup <= 2 and detour_extra <= 1:
                return True
            return via_pickup <= 2 and detour_extra <= 3 and pickup_order.p >= delivery_order.p

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
        pickup_score = self._pickup_key(shipper, pickup_order, orders, t)
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
    # Action planning
    # ------------------------------------------------------------------
    def _record_recent_pickups(self, obs: dict) -> None:
        if not (13 <= self._map_size < 18 and self._shipper_count >= 3):
            return
        orders: Dict[int, Order] = obs["orders"]
        for oid in obs.get("new_order_ids", []):
            order = orders.get(oid)
            if order is not None:
                self._recent_pickups.append((order.sx, order.sy))

    def _idle_goal(self, shipper: Shipper) -> Optional[Position]:
        if not self._recent_pickups:
            return None

        recent_window = list(self._recent_pickups)
        if self._shipper_count <= 3:
            recent_window = recent_window[-10:]
        counts = Counter(recent_window)
        best_goal: Optional[Position] = None
        best_key: Optional[Tuple[float, int, Position]] = None
        for pos, count in counts.items():
            distance = self._distance(shipper.position, pos)
            if distance <= 0 or distance >= INF:
                continue
            key = (-count / max(distance, 1), distance, pos)
            if best_key is None or key < best_key:
                best_key = key
                best_goal = pos
        return best_goal

    def _planned_action(self, shipper: Shipper, goal: Position, op_at_goal: int) -> Tuple[Action, Position]:
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        op = op_at_goal if next_position == goal else 0
        return (move, op), next_position

    def _raw_actions(self, obs: dict) -> Dict[int, Tuple[Action, Position]]:
        if self.grid is not obs["grid"]:
            self.grid = obs["grid"]
            self._path_cache.clear()
        self._map_size = int(obs.get("N", len(self.grid)))
        self._shipper_count = int(obs.get("C", len(obs.get("shippers", []))))
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))

        self._record_recent_pickups(obs)

        planned: Dict[int, Tuple[Action, Position]] = {}
        reserved_pickups: Set[int] = set()

        if 12 <= self._map_size < 18:
            shipper_order = sorted(
                shippers,
                key=lambda s: (
                    min((self._delivery_key(s, o, t) for o in self._carried_orders(s, orders)), default=(1, INF, INF, 0, s.id)),
                    s.id,
                ),
            )
        else:
            shipper_order = sorted(shippers, key=lambda s: s.id)

        for shipper in shipper_order:
            delivery_order = self._select_delivery(shipper, orders, t)
            pickup_order = self._select_pickup(shipper, orders, reserved_pickups, t)

            if delivery_order is not None:
                if (
                    pickup_order is not None
                    and self._should_pickup_before_delivery(shipper, pickup_order, delivery_order, orders, t)
                ):
                    reserved_pickups.add(pickup_order.id)
                    planned[shipper.id] = self._planned_action(shipper, (pickup_order.sx, pickup_order.sy), 1)
                    continue

                planned[shipper.id] = self._planned_action(shipper, (delivery_order.ex, delivery_order.ey), 2)
                continue

            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                planned[shipper.id] = self._planned_action(shipper, (pickup_order.sx, pickup_order.sy), 1)
                continue

            idle_goal = self._idle_goal(shipper)
            if idle_goal is not None:
                planned[shipper.id] = self._planned_action(shipper, idle_goal, 0)
                continue

            planned[shipper.id] = (("S", 0), shipper.position)

        return planned

    def _avoid_conflicts(
        self,
        shippers: List[Shipper],
        planned: Dict[int, Tuple[Action, Position]],
    ) -> Dict[int, Action]:
        actions: Dict[int, Action] = {}
        old_positions = {shipper.id: shipper.position for shipper in shippers}
        occupied = set(old_positions.values())

        for shipper in sorted(shippers, key=lambda s: s.id):
            action, target = planned.get(shipper.id, (("S", 0), shipper.position))
            occupied.discard(old_positions[shipper.id])

            if target in occupied:
                actions[shipper.id] = ("S", 0)
                occupied.add(old_positions[shipper.id])
                continue

            actions[shipper.id] = action
            occupied.add(target)

        return actions

    def _decide_actions_cbs(self, obs: dict) -> Dict[int, Action]:
        planned = self._raw_actions(obs)
        if int(obs.get("N", 0)) < 18:
            return {sid: action for sid, (action, _) in planned.items()}
        return self._avoid_conflicts(obs["shippers"], planned)

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions_cbs(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
