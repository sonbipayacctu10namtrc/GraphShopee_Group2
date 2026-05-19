from __future__ import annotations

import math
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


class ACOSolver(Solver):
    """Ant Colony Optimization policy cho môi trường Online MAPD."""

    method_name = "ACO"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._pheromone: Dict[Tuple[str, int], float] = {}
        self._alpha = 1.2
        self._beta = 2.0
        self._rho = 0.08
        self._deposit = 0.8
        self._min_pheromone = 0.15
        self._max_pheromone = 6.0

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
    # ACO scoring
    # ------------------------------------------------------------------
    def _pheromone_key(self, task_type: str, order: Order) -> Tuple[str, int]:
        return task_type, order.id

    def _get_pheromone(self, task_type: str, order: Order) -> float:
        return self._pheromone.get(self._pheromone_key(task_type, order), 1.0)

    def _evaporate_pheromone(self) -> None:
        for key, value in list(self._pheromone.items()):
            value = max(self._min_pheromone, value * (1.0 - self._rho))
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

    def _estimated_reward(self, order: Order, finish_t: int) -> float:
        return max(0.0, delivery_reward(order, finish_t, int(self.cfg.get("T", 1))))

    def _reinforcement_amount(self, base_amount: float, reward: float) -> float:
        reward_scale = 1.0 + min(1.0, reward / 25.0)
        return base_amount * reward_scale

    def _delivery_heuristic(self, shipper: Shipper, order: Order, t: int) -> float:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        if distance >= INF:
            return 0.0

        finish_t = t + distance
        slack = order.et - finish_t
        urgency = 1.0 / (1.0 + max(slack, 0))
        late_penalty = 0.25 if finish_t > order.et else 1.0
        priority = 1.0 + 0.45 * order.p
        reward_factor = 1.0 + min(2.0, self._estimated_reward(order, finish_t) / 20.0)
        return late_penalty * priority * reward_factor * (1.0 + urgency) / (1.0 + distance)

    def _pickup_heuristic(self, shipper: Shipper, order: Order, t: int) -> float:
        pickup_pos = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup_pos)
        delivery_distance = self._distance(pickup_pos, delivery_pos)
        if pickup_distance >= INF or delivery_distance >= INF:
            return 0.0

        finish_t = t + pickup_distance + delivery_distance
        slack = order.et - finish_t
        late_penalty = 0.25 if finish_t > order.et else 1.0
        priority = 1.0 + 0.5 * order.p
        capacity_fit = max(0.2, 1.0 - order.w / max(shipper.W_max, 1.0))
        urgency = 1.0 / (1.0 + max(slack, 0))
        travel = 1.0 + pickup_distance + 0.35 * delivery_distance
        reward_factor = 1.0 + min(2.5, self._estimated_reward(order, finish_t) / 20.0)
        return late_penalty * priority * reward_factor * capacity_fit * (1.0 + 0.4 * urgency) / travel

    def _delivery_score(self, shipper: Shipper, order: Order, t: int) -> float:
        return self._aco_value(
            self._get_pheromone("deliver", order),
            self._delivery_heuristic(shipper, order, t),
        )

    def _pickup_score(self, shipper: Shipper, order: Order, t: int) -> float:
        return self._aco_value(
            self._get_pheromone("pickup", order),
            self._pickup_heuristic(shipper, order, t),
        )

    def _delivery_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, float, int, float, int]:
        distance = self._distance(shipper.position, (order.ex, order.ey))
        finish_t = t + distance
        slack = order.et - finish_t
        estimated_reward = self._estimated_reward(order, finish_t)
        return (
            1 if finish_t > order.et else 0,
            slack,
            distance,
            -estimated_reward,
            -order.p,
            -self._get_pheromone("deliver", order),
            order.id,
        )

    def _pickup_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int, float, int]:
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
            -self._get_pheromone("pickup", order),
            order.id,
        )

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Order]:
        candidates = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda order: self._delivery_rank(shipper, order, t),
        )

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
            if self._pickup_heuristic(shipper, order, t) <= 0.0:
                continue
            candidates.append(order)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda order: self._pickup_rank(shipper, order, t),
        )

    def _assign_pickups(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        t: int,
    ) -> Dict[int, Order]:
        candidates: List[Tuple[Tuple[int, int, int, int, int, float, int], float, int, Order]] = []

        for shipper in shippers:
            for order in orders.values():
                if not shipper.can_carry(order, orders):
                    continue
                if self._pickup_heuristic(shipper, order, t) <= 0.0:
                    continue
                candidates.append(
                    (
                        self._pickup_rank(shipper, order, t),
                        -self._pickup_score(shipper, order, t),
                        shipper.id,
                        order,
                    )
                )

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].id))

        assignments: Dict[int, Order] = {}
        used_shippers: set[int] = set()
        used_orders: set[int] = set()

        for _, _, shipper_id, order in candidates:
            if shipper_id in used_shippers or order.id in used_orders:
                continue
            used_shippers.add(shipper_id)
            used_orders.add(order.id)
            assignments[shipper_id] = order

        return assignments

    def _should_pickup_before_delivery(
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

        delivery_pos = (delivery_order.ex, delivery_order.ey)
        pickup_pos = (pickup_order.sx, pickup_order.sy)
        direct_finish = t + self._distance(shipper.position, delivery_pos)
        detour_finish = (
            t
            + self._distance(shipper.position, pickup_pos)
            + self._distance(pickup_pos, delivery_pos)
        )
        if detour_finish >= INF:
            return False
        if direct_finish <= delivery_order.et < detour_finish:
            return False

        pickup_rank = self._pickup_rank(shipper, pickup_order, t)
        delivery_rank = self._delivery_rank(shipper, delivery_order, t)
        detour_extra = detour_finish - direct_finish
        pickup_finish = (
            t
            + self._distance(shipper.position, pickup_pos)
            + self._distance(pickup_pos, (pickup_order.ex, pickup_order.ey))
        )
        pickup_reward = self._estimated_reward(pickup_order, pickup_finish)
        extra_allowance = 1 if pickup_order.p >= 2 and pickup_reward >= 20.0 else 0
        return (
            detour_extra <= 3 + extra_allowance
            and pickup_rank[0] == 0
            and (
                pickup_order.p > delivery_order.p
                or pickup_rank[1] <= max(2, delivery_rank[2])
                or self._pickup_score(shipper, pickup_order, t) > self._delivery_score(shipper, delivery_order, t) * 1.5
            )
        )

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

        delivery_orders = {
            shipper.id: self._select_delivery(shipper, orders, t)
            for shipper in shippers
        }
        idle_shippers = [shipper for shipper in shippers if delivery_orders[shipper.id] is None]

        actions: Dict[int, Action] = {}
        reserved_pickups: set[int] = set()
        assigned_pickups = self._assign_pickups(idle_shippers, orders, t)
        reserved_pickups.update(order.id for order in assigned_pickups.values())

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = delivery_orders[shipper.id]
            pickup_order = assigned_pickups.get(shipper.id)

            if delivery_order is not None:
                pickup_order = self._select_pickup(shipper, orders, reserved_pickups, t)

            if delivery_order is not None:
                if (
                    pickup_order is not None
                    and self._should_pickup_before_delivery(shipper, pickup_order, delivery_order, t)
                ):
                    reserved_pickups.add(pickup_order.id)
                    pickup_finish = (
                        t
                        + self._distance(shipper.position, (pickup_order.sx, pickup_order.sy))
                        + self._distance((pickup_order.sx, pickup_order.sy), (pickup_order.ex, pickup_order.ey))
                    )
                    pickup_reward = self._estimated_reward(pickup_order, pickup_finish)
                    self._reinforce(
                        "pickup",
                        pickup_order,
                        self._reinforcement_amount(self._deposit * 0.3, pickup_reward),
                    )
                    actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                    continue

                delivery_finish = t + self._distance(shipper.position, (delivery_order.ex, delivery_order.ey))
                delivery_reward_value = self._estimated_reward(delivery_order, delivery_finish)
                self._reinforce(
                    "deliver",
                    delivery_order,
                    self._reinforcement_amount(self._deposit * 0.5, delivery_reward_value),
                )
                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                pickup_finish = (
                    t
                    + self._distance(shipper.position, (pickup_order.sx, pickup_order.sy))
                    + self._distance((pickup_order.sx, pickup_order.sy), (pickup_order.ex, pickup_order.ey))
                )
                pickup_reward = self._estimated_reward(pickup_order, pickup_finish)
                self._reinforce(
                    "pickup",
                    pickup_order,
                    self._reinforcement_amount(self._deposit * 0.4, pickup_reward),
                )
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
            self._evaporate_pheromone()
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
