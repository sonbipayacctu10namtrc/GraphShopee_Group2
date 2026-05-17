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


class VRPOrToolsSolver(Solver):
    """
    VRP-style online solver.

    OR-Tools is not available in the provided environment, so this class uses
    the same online env API with a small dynamic VRP heuristic:
    - assign visible pickup tasks to shippers by marginal route cost;
    - prioritize carried deliveries by deadline feasibility and priority;
    - move one BFS step per simulation tick.
    """

    method_name = "VRP-OrTools"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}

    # ------------------------------------------------------------------
    # Grid/BFS utilities
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
    # VRP scoring
    # ------------------------------------------------------------------
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
        return min(self._distance(shipper.position, (o.ex, o.ey)) for o in carried)

    def _delivery_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int]:
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

    def _pickup_rank(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> Tuple[int, int, int, int, int, int, int]:
        pickup_pos = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup_pos)
        delivery_distance = self._distance(pickup_pos, delivery_pos)
        finish_t = t + pickup_distance + delivery_distance
        slack = order.et - finish_t

        # If the shipper already carries packages, treat pickup as a VRP
        # insertion before the nearest current delivery.
        nearest_delivery = self._carried_delivery_cost(shipper, orders)
        insertion_penalty = 0
        if nearest_delivery > 0:
            carried = self._select_delivery(shipper, orders, t)
            if carried is not None:
                current_delivery_pos = (carried.ex, carried.ey)
                insertion_penalty = (
                    pickup_distance
                    + self._distance(pickup_pos, current_delivery_pos)
                    - nearest_delivery
                )

        return (
            1 if finish_t > order.et else 0,
            insertion_penalty,
            pickup_distance,
            -order.p,
            slack,
            delivery_distance,
            order.id,
        )

    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Order]:
        candidates = self._carried_orders(shipper, orders)
        if not candidates:
            return None
        return min(candidates, key=lambda order: self._delivery_rank(shipper, order, t))

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
        t: int,
    ) -> Optional[Order]:
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            if self._distance(shipper.position, (order.sx, order.sy)) >= INF:
                continue
            if self._distance((order.sx, order.sy), (order.ex, order.ey)) >= INF:
                continue
            candidates.append(order)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda order: self._pickup_rank(shipper, order, orders, t),
        )

    def _can_insert_pickup(
        self,
        shipper: Shipper,
        pickup_order: Order,
        delivery_order: Order,
        t: int,
    ) -> bool:
        if len(shipper.bag) >= shipper.K_max:
            return False
        if int(self.cfg.get("N", 0)) < 12:
            return False

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

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        move = self._next_move(shipper.position, goal)
        return move, valid_next_pos(shipper.position, move, self.grid)

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 1) if next_position == goal else (move, 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs.get("t", 0))

        actions: Dict[int, Action] = {}
        reserved_pickups: set[int] = set()

        # Dispatch shippers with tighter carried deliveries first. This mimics
        # a small multi-vehicle VRP assignment at each online decision point.
        shipper_order = sorted(
            shippers,
            key=lambda s: (
                min((self._delivery_rank(s, o, t) for o in self._carried_orders(s, orders)), default=(1, INF, INF, 0, s.id)),
                s.id,
            ),
        )

        for shipper in shipper_order:
            delivery_order = self._select_delivery(shipper, orders, t)
            pickup_order = self._select_pickup(shipper, orders, reserved_pickups, t)

            if delivery_order is not None:
                if pickup_order is not None and self._can_insert_pickup(shipper, pickup_order, delivery_order, t):
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
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
