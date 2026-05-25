from __future__ import annotations

import time
from collections import deque
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
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}

    # ------------------------------------------------------------------
    # Grid search
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

    def _distance(self, start: Position, goal: Position) -> int:
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

    def _delivery_key(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int]:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        finish_t = t + distance
        return (
            1 if finish_t > order.et else 0,
            order.et - finish_t,
            distance,
            -order.p,
            order.id,
        )

    def _pickup_key(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int, int]:
        pickup = (order.sx, order.sy)
        delivery = (order.ex, order.ey)
        pickup_dist = self._distance(shipper.position, pickup)
        delivery_dist = self._distance(pickup, delivery)
        finish_t = t + pickup_dist + delivery_dist
        return (
            1 if finish_t > order.et else 0,
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

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        t: int,
    ) -> Optional[Order]:
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids or not self._can_carry(shipper, order, orders):
                continue
            pickup_dist = self._distance(shipper.position, (order.sx, order.sy))
            delivery_dist = self._distance((order.sx, order.sy), (order.ex, order.ey))
            if pickup_dist >= INF or delivery_dist >= INF:
                continue
            candidates.append(order)

        if not candidates:
            return None
        return min(candidates, key=lambda order: self._pickup_key(shipper, order, t))

    def _should_pickup_before_delivery(
        self,
        shipper: Shipper,
        pickup_order: Order,
        delivery_order: Order,
        t: int,
    ) -> bool:
        map_size = int(self.cfg.get("N", 0))
        if map_size < 12:
            return False
        if len(shipper.bag) >= shipper.K_max:
            return False

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

            return via_pickup <= 2 and detour_finish - direct_finish <= 3 and pickup_order.p >= delivery_order.p

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
    # Action planning
    # ------------------------------------------------------------------
    def _planned_action(self, shipper: Shipper, goal: Position, op_at_goal: int) -> Tuple[Action, Position]:
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        op = op_at_goal if next_position == goal else 0
        return (move, op), next_position

    def _raw_actions(self, obs: dict) -> Dict[int, Tuple[Action, Position]]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))

        planned: Dict[int, Tuple[Action, Position]] = {}
        reserved_pickups: Set[int] = set()

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = self._select_delivery(shipper, orders, t)
            pickup_order = self._select_pickup(shipper, orders, reserved_pickups, t)

            if delivery_order is not None:
                if (
                    pickup_order is not None
                    and self._should_pickup_before_delivery(shipper, pickup_order, delivery_order, t)
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
