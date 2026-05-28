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
    """Current task target of one shipper: pickup or deliver one order at pos."""

    kind: TaskKind
    order_id: int
    pos: Position


class Constraint(NamedTuple):
    """CBS constraint: forbid a vertex at time, or an edge transition at time."""

    shipper_id: int
    kind: str
    pos: Position
    time: int
    to_pos: Optional[Position] = None


class _FlowEdge:
    """Residual edge used by min-cost flow assignment."""

    def __init__(self, to: int, rev: int, cap: int, cost: int) -> None:
        """Store residual endpoint, reverse-edge index, capacity and edge cost."""
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost


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
        """Initialize caches, online hotspot/surge state, and adaptive mode flags."""
        super().__init__(env)
        self.cfg = {"N": env.N, "C": env.C, "G": env.G, "T": env.T, "name": env.config_name}
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
        self._new_order_history: deque[Tuple[int, int]] = deque()
        self._surge_score = 0.0
        self._active_pressure = 0.0
        self._sticky_pickups: Dict[int, Tuple[int, int, float]] = {}

    def _refresh_from_obs(self, obs: dict) -> None:
        """
        Refresh public instance parameters from observation only.

        Large mode is enabled by C >= 12 or G >= 500; the planning window is
        clamped to [12, 34]. If the grid object changes, cached paths/distances
        are invalidated.
        """
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
        self._planning_window = max(12, min(34, self._n + 8))

    # ------------------------------------------------------------------
    # Grid utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position, include_wait: bool = False) -> Iterable[Tuple[Move, Position]]:
        """Yield valid next cells; include "S" only when waiting is useful for CBS/A*."""
        moves = ALL_MOVES if include_wait else MOVES
        for move in moves:
            nxt = valid_next_pos(pos, move, self.grid)
            if include_wait or nxt != pos:
                yield move, nxt

    def _bfs_path(self, start: Position, goal: Position) -> Optional[List[Position]]:
        """
        Compute exact shortest path on the grid with BFS and cache it.

        The resulting distance is len(path)-1. This is used both for assignment
        scoring and as an admissible heuristic inside constrained A*.
        """
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
        """Return cached BFS shortest-path distance, or INF if unreachable."""
        if start == goal:
            return 0
        key = (start, goal)
        if key not in self._distance_cache:
            path = self._bfs_path(start, goal)
            self._distance_cache[key] = INF if path is None else len(path) - 1
        return self._distance_cache[key]

    def _manhattan(self, start: Position, goal: Position) -> int:
        """Fast lower-bound distance |dr|+|dc| used mainly for large-mode pruning."""
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def _move_between(self, start: Position, nxt: Position) -> Move:
        """Convert two adjacent cells into an env move; return "S" if unchanged/invalid."""
        if start == nxt:
            return "S"
        for move, pos in self._neighbors(start):
            if pos == nxt:
                return move
        return "S"

    def _next_move(self, start: Position, goal: Position) -> Move:
        """Return the first move of a cached BFS path from start to goal."""
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
        """List undelivered orders currently carried by a shipper."""
        return [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

    def _delivery_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[int, int, int, int, int]:
        """
        Rank carried orders for delivery in small/medium mode.

        Formula:
          finish_t = t + d(shipper, dropoff)
          rps = delivery_reward(order, finish_t, T) / max(d, 1)
        Sort favors reachable, higher reward-per-step, on-time, shorter distance,
        higher priority, then lower order id.
        """
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
        """
        Estimate pickup hotspot value from public revealed orders.

        Formula:
          bonus = hotspot_score(pickup) * (1 + 0.20 * surge_score)
        The bonus is disabled on tiny maps or low-shipper large maps where it
        caused unstable over-biasing.
        """
        n = self._n
        c = self._c
        if n < 18 or (n >= 30 and c <= 3):
            return 0.0
        return self._hotspot_tracker.score((order.sx, order.sy)) * (1.0 + 0.20 * self._surge_score)

    def _update_surge_score(self, obs: dict) -> None:
        """
        Infer surge intensity from recent public new_order_ids.

        Formula:
          recent_rate = new_orders_in_last_60_steps / 60
          base_rate = G / T
          surge_score = clamp((recent_rate/base_rate - 1) / 3, 0, 1)
        """
        t = int(obs.get("t", 0))
        self._new_order_history.append((t, len(obs.get("new_order_ids", []))))
        window = 60
        while self._new_order_history and t - self._new_order_history[0][0] > window:
            self._new_order_history.popleft()
        recent_rate = sum(count for _, count in self._new_order_history) / max(1, window)
        base_rate = self._g / max(1, self._t_limit)
        self._surge_score = max(0.0, min(1.0, (recent_rate / max(base_rate, 1e-6) - 1.0) / 3.0))

    def _region_weight(self) -> float:
        """
        Compute region penalty weight for the N=18 case.

        Formula:
          obstacle_ratio = blocked / (N*N)
          free_per_shipper = free / C
          weight = 0.26 + obstacle_bonus + spacious_bonus, capped at 0.44
        """
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
        """Map a position to one of four quadrants: 0, 1, 2, or 3."""
        n = max(1, self._n)
        r, c = pos
        return (0 if r < n // 2 else 2) + (0 if c < n // 2 else 1)

    def _region_penalty(self, shipper: Shipper, order: Order) -> float:
        """
        Penalize assigning an order outside the shipper's current quadrant.

        Formula:
          penalty = weight * (1.0 if pickup region differs else 0
                              + 0.35 if dropoff region differs else 0)
        """
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
        """
        Rank pickup candidates in small/medium mode.

        Formula:
          route = d(shipper,pickup) + d(pickup,dropoff)
          finish_t = t + route
          rps = delivery_reward(order, finish_t, T) / max(route, 1)
          rps *= 1 + 0.10 * hotspot_bonus
        Also adds insertion_penalty if picking this order delays the best carried
        delivery, and region_penalty if it pulls the shipper across regions.
        """
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
        """
        Score one order from an arbitrary start position for beam lookahead.

        Formula:
          value = reward * (1 + 0.10*hotspot)
                  - 0.06*route_distance - 0.12*lateness + 0.8*priority
        """
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
        Look ahead after the first candidate order on N=12 maps.

        Formula:
          lookahead = first_value + 0.35 * sum(top_2_followup_values)
        The lookahead term is inserted into the pickup rank only for choosing the
        first target; CBS still plans the actual step path afterward.
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
        """Check capacity/weight and reachability for shipper -> pickup -> dropoff."""
        if not shipper.can_carry(order, orders):
            return False
        return (
            self._distance(shipper.position, (order.sx, order.sy)) < INF
            and self._distance((order.sx, order.sy), (order.ex, order.ey)) < INF
        )

    def _expected_pickup_reward(self, shipper: Shipper, order: Order, t: int) -> float:
        """Return delivery_reward at t + d(shipper,pickup) + d(pickup,dropoff)."""
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup)
        delivery_distance = self._distance(pickup, dropoff)
        if pickup_distance >= INF or delivery_distance >= INF:
            return 0.0
        return delivery_reward(order, t + pickup_distance + delivery_distance, self._t_limit)

    def _large_delivery_rank(self, shipper: Shipper, order: Order, t: int) -> Tuple[float, ...]:
        """
        Rank carried orders in large mode.

        Formula:
          finish_t = t + d(shipper, dropoff)
          score starts with -reward/max(distance,1), then lateness/slack/distance.
        """
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
        """
        Cheaply shortlist orders in large mode using Manhattan distance.

        Formula:
          route = manhattan(shipper,pickup) + manhattan(pickup,dropoff)
          finish_t = t + route
        Depending on pressure, sort by reward-per-step or absolute reward.
        """
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._manhattan(shipper.position, pickup)
        delivery_distance = self._manhattan(pickup, dropoff)
        route_distance = pickup_distance + delivery_distance
        finish_t = t + route_distance
        reward = delivery_reward(order, finish_t, self._t_limit)
        if self._use_reward_per_step_prefilter():
            reward_per_step = reward / max(route_distance, 1)
            return (
                1 if reward <= 0.0 else 0,
                1 if finish_t > order.et else 0,
                -reward_per_step,
                pickup_distance,
                delivery_distance,
                order.et,
                -order.p,
                order.id,
            )
        return (
            -reward,
            pickup_distance,
            1 if finish_t > order.et else 0,
            order.et,
            -order.p,
            order.id,
        )

    def _use_reward_per_step_prefilter(self) -> bool:
        """Use reward-per-step prefilter on large/high-pressure/surge instances."""
        if self._n < 55 and self._c <= 12 and self._active_pressure < 8.0:
            return False
        return self._n >= 55 or self._active_pressure >= 8.0 or self._surge_score >= 0.55

    def _large_pickup_rank(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> Tuple[float, ...]:
        """
        Rank pickup candidates in large mode with Manhattan estimates.

        Formula:
          rps = delivery_reward(order, t+pd+dd, T) / max(pd+dd, 1)
          rps *= 1 + 0.08 * hotspot_bonus
        Includes insertion penalty for delaying carried deliveries.
        """
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

    def _large_pickup_value(self, shipper: Shipper, order: Order, t: int) -> float:
        """
        Scalar value for large-mode pickup and sticky comparison.

        Formula:
          value = rps*(1+0.08*hotspot) + 0.015*reward + 0.12*priority
                  - 0.015*pickup_distance - 0.020*lateness
        """
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._manhattan(shipper.position, pickup)
        delivery_distance = self._manhattan(pickup, dropoff)
        route_distance = pickup_distance + delivery_distance
        finish_t = t + route_distance
        reward = delivery_reward(order, finish_t, self._t_limit)
        reward_per_step = reward / max(route_distance, 1)
        lateness = max(0, finish_t - order.et)
        return (
            reward_per_step * (1.0 + 0.08 * self._hotspot_bonus(order))
            + 0.015 * reward
            + 0.12 * order.p
            - 0.015 * pickup_distance
            - 0.020 * lateness
        )

    def _should_use_flow_assignment(self, active_orders_count: int) -> bool:
        """
        Decide when global min-cost flow matching is worth using.

        Formula:
          avg_route_len = 4N/3
          budget_per_order = T*C/G
          difficulty = avg_route_len / budget_per_order
          active_pressure = active_orders/C

        Flow is intentionally conservative. It improves max/bottleneck cases,
        but on medium-large maps with many shippers it can over-coordinate and
        reduce delivered orders. The rule therefore keeps greedy assignment for
        those cases and enables flow only for very hard/max, bottleneck-like, or
        severe-queue situations.
        """
        c = max(1, self._c)
        g = max(1, self._g)
        t_limit = max(1, self._t_limit)

        total_cells = max(1, self._n * self._n)
        blocked = sum(cell == 1 for row in self.grid for cell in row)
        obstacle_ratio = blocked / total_cells

        avg_route_len = 4.0 * self._n / 3.0
        budget_per_order = t_limit * c / g
        difficulty = avg_route_len / max(1.0, budget_per_order)
        active_pressure = active_orders_count / c
        free_per_shipper = (total_cells - blocked) / c

        very_hard = difficulty >= 3.0
        bottleneck_like = difficulty >= 2.0 and c <= 12
        sparse_large = difficulty >= 2.6 and free_per_shipper >= 450 and obstacle_ratio <= 0.08
        severe_queue = active_pressure >= 8.0 and (very_hard or c <= 12)

        return very_hard or bottleneck_like or sparse_large or severe_queue

    def _add_flow_edge(self, graph: List[List[_FlowEdge]], fr: int, to: int, cap: int, cost: int) -> _FlowEdge:
        """Add forward and reverse residual edges for min-cost flow."""
        forward = _FlowEdge(to, len(graph[to]), cap, cost)
        backward = _FlowEdge(fr, len(graph[fr]), 0, -cost)
        graph[fr].append(forward)
        graph[to].append(backward)
        return forward

    def _min_cost_positive_matching(self, candidates: List[Tuple[int, int, float]]) -> Dict[int, int]:
        """
        Solve a positive-value bipartite assignment with min-cost flow.

        Graph:
          source -> shipper -> order -> sink
        Each shipper/order has capacity 1. Edge cost is -1000*value, so a
        shortest augmenting path with negative total cost increases total value.
        Augmentation stops when no negative-cost path remains.
        """
        shipper_ids = sorted({sid for sid, _, value in candidates if value > 0.0})
        order_ids = sorted({oid for _, oid, value in candidates if value > 0.0})
        if not shipper_ids or not order_ids:
            return {}

        shipper_index = {sid: idx for idx, sid in enumerate(shipper_ids)}
        order_index = {oid: idx for idx, oid in enumerate(order_ids)}

        source = 0
        shipper_offset = 1
        order_offset = shipper_offset + len(shipper_ids)
        sink = order_offset + len(order_ids)
        graph: List[List[_FlowEdge]] = [[] for _ in range(sink + 1)]

        for sid in shipper_ids:
            self._add_flow_edge(graph, source, shipper_offset + shipper_index[sid], 1, 0)
        for oid in order_ids:
            self._add_flow_edge(graph, order_offset + order_index[oid], sink, 1, 0)

        edge_meta: List[Tuple[_FlowEdge, int, int]] = []
        best_edge: Dict[Tuple[int, int], float] = {}
        for sid, oid, value in candidates:
            if value <= 0.0:
                continue
            key = (sid, oid)
            if value > best_edge.get(key, -1.0):
                best_edge[key] = value

        for (sid, oid), value in best_edge.items():
            fr = shipper_offset + shipper_index[sid]
            to = order_offset + order_index[oid]
            edge = self._add_flow_edge(graph, fr, to, 1, int(round(-1000.0 * value)))
            edge_meta.append((edge, sid, oid))

        max_flow = min(len(shipper_ids), len(order_ids))
        for _ in range(max_flow):
            dist = [10**18] * len(graph)
            in_queue = [False] * len(graph)
            prev_node = [-1] * len(graph)
            prev_edge = [-1] * len(graph)
            dist[source] = 0
            queue: deque[int] = deque([source])
            in_queue[source] = True

            while queue:
                node = queue.popleft()
                in_queue[node] = False
                for edge_idx, edge in enumerate(graph[node]):
                    if edge.cap <= 0:
                        continue
                    new_dist = dist[node] + edge.cost
                    if new_dist >= dist[edge.to]:
                        continue
                    dist[edge.to] = new_dist
                    prev_node[edge.to] = node
                    prev_edge[edge.to] = edge_idx
                    if not in_queue[edge.to]:
                        queue.append(edge.to)
                        in_queue[edge.to] = True

            if prev_node[sink] == -1 or dist[sink] >= 0:
                break

            node = sink
            while node != source:
                parent = prev_node[node]
                edge = graph[parent][prev_edge[node]]
                edge.cap -= 1
                graph[node][edge.rev].cap += 1
                node = parent

        assignments: Dict[int, int] = {}
        for edge, sid, oid in edge_meta:
            if edge.cap == 0:
                assignments[sid] = oid
        return assignments

    def _pickup_flow_value(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> float:
        """
        Edge value for flow assignment between one shipper and one order.

        Formula:
          value = 10*rps*hotspot_factor + 0.03*reward + 0.50*priority
                  - 0.08*pickup_distance - 0.04*delivery_distance
                  - 0.20*lateness - insertion_penalty - 0.60*region_penalty
        """
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        pickup_distance = self._distance(shipper.position, pickup)
        delivery_distance = self._distance(pickup, dropoff)
        if pickup_distance >= INF or delivery_distance >= INF:
            return -1.0

        route_distance = pickup_distance + delivery_distance
        finish_t = t + route_distance
        reward = delivery_reward(order, finish_t, self._t_limit)
        if reward <= 0.0:
            return -1.0

        carried = self._carried_orders(shipper, orders)
        insertion_penalty = 0.0
        if carried:
            best_delivery = min(carried, key=lambda o: self._delivery_rank(shipper, o, t))
            direct = self._distance(shipper.position, (best_delivery.ex, best_delivery.ey))
            via_pickup = pickup_distance + self._distance(pickup, (best_delivery.ex, best_delivery.ey))
            insertion_penalty = max(0.0, float(via_pickup - direct))

        lateness = max(0, finish_t - order.et)
        reward_per_step = reward / max(route_distance, 1)
        hotspot_factor = 1.0 + 0.10 * self._hotspot_bonus(order)
        return (
            10.0 * reward_per_step * hotspot_factor
            + 0.03 * reward
            + 0.50 * order.p
            - 0.08 * pickup_distance
            - 0.04 * delivery_distance
            - 0.20 * lateness
            - 1.00 * insertion_penalty
            - 0.60 * self._region_penalty(shipper, order)
        )

    def _apply_flow_pickups(
        self,
        targets: Dict[int, Target],
        orders: Dict[int, Order],
        candidates: List[Tuple[int, int, float]],
    ) -> Dict[int, Target]:
        """Convert min-cost-flow assignments into pickup Targets without replacing deliveries."""
        assignments = self._min_cost_positive_matching(candidates)
        for shipper_id, order_id in assignments.items():
            if shipper_id in targets:
                continue
            order = orders.get(order_id)
            if order is None or order.picked or order.delivered:
                continue
            targets[shipper_id] = Target("pickup", order.id, (order.sx, order.sy))
        return targets

    def _apply_sticky_pickups(
        self,
        targets: Dict[int, Target],
        shippers: List[Shipper],
        orders: Dict[int, Order],
        t: int,
    ) -> Dict[int, Target]:
        """
        Keep a previous pickup target unless a new one is clearly better.

        Formula:
          keep old target if current_value <= max(old_value, sticky_value) * 1.25
        Sticky expires after 8 timesteps or when the order is no longer valid.
        """
        max_age = 8
        keep_margin = 1.25
        used_orders = {
            target.order_id
            for target in targets.values()
            if target.kind == "pickup"
        }

        for shipper in shippers:
            target = targets.get(shipper.id)
            if target is not None and target.kind == "deliver":
                self._sticky_pickups.pop(shipper.id, None)
                continue

            sticky = self._sticky_pickups.get(shipper.id)
            if sticky is None:
                continue
            old_order_id, assigned_t, old_value = sticky
            old_order = orders.get(old_order_id)
            if (
                old_order is None
                or old_order.picked
                or old_order.delivered
                or t - assigned_t > max_age
                or not shipper.can_carry(old_order, orders)
            ):
                self._sticky_pickups.pop(shipper.id, None)
                continue

            current_value = -1e9
            if target is not None and target.kind == "pickup":
                current_order = orders.get(target.order_id)
                if current_order is not None:
                    current_value = self._large_pickup_value(shipper, current_order, t)

            sticky_value = self._large_pickup_value(shipper, old_order, t)
            old_taken_by_other = old_order_id in used_orders and (target is None or target.order_id != old_order_id)
            if old_taken_by_other:
                continue
            if target is None or current_value <= max(old_value, sticky_value) * keep_margin:
                if target is not None and target.kind == "pickup":
                    used_orders.discard(target.order_id)
                targets[shipper.id] = Target("pickup", old_order.id, (old_order.sx, old_order.sy))
                used_orders.add(old_order.id)

        refreshed: Dict[int, Tuple[int, int, float]] = {}
        for shipper in shippers:
            target = targets.get(shipper.id)
            if target is None or target.kind != "pickup":
                continue
            order = orders.get(target.order_id)
            if order is None:
                continue
            previous = self._sticky_pickups.get(shipper.id)
            assigned_t = previous[1] if previous is not None and previous[0] == order.id else t
            refreshed[shipper.id] = (order.id, assigned_t, self._large_pickup_value(shipper, order, t))
        self._sticky_pickups = refreshed
        return targets

    def _assign_targets_large(self, obs: dict) -> Dict[int, Target]:
        """
        Assign deliver/pickup targets in large mode.

        Delivery targets are fixed first for carried orders. Pickup candidates
        are shortlisted, then either greedily selected or globally matched with
        min-cost flow when _should_use_flow_assignment is true. Sticky pickup is
        applied last to reduce target switching.
        """
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
        self._active_pressure = len(active_orders) / max(1, self._c)
        if not active_orders:
            return targets
        use_flow = self._should_use_flow_assignment(len(active_orders))

        candidate_limit = max(18, min(70, 3 * max(1, self._c)))
        exact_limit = max(3, min(5, self._c // 2))
        if self._n >= 70 or self._g >= 1000:
            candidate_limit = max(16, min(36, 2 * max(1, self._c)))
            exact_limit = 1 if self._active_pressure <= 3.0 and self._n < 90 else 0
        elif self._active_pressure >= 8.0:
            candidate_limit = max(18, min(54, 2 * max(1, self._c)))
            exact_limit = max(2, min(4, self._c // 4))

        pickup_pairs: List[Tuple[Tuple[float, ...], int, Order]] = []
        for shipper in shippers:
            if (self._n >= 70 or self._g >= 1000) and shipper.id in targets:
                continue
            if shipper.id in targets and len(shipper.bag) >= max(1, shipper.K_max):
                continue
            current_weight = sum(orders[oid].w for oid in shipper.bag if oid in orders)
            remaining_weight = shipper.W_max - current_weight
            remaining_slots = shipper.K_max - len(shipper.bag)
            if remaining_slots <= 0 or remaining_weight <= 0:
                continue
            feasible = [order for order in active_orders if order.w <= remaining_weight]
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
                if use_flow:
                    value = (
                        self._pickup_flow_value(shipper, order, orders, t)
                        if exact_limit > 0
                        else self._large_pickup_value(shipper, order, t)
                    )
                    if value > 0.0:
                        pickup_pairs.append(((0.0, -value), shipper.id, order))
                else:
                    pickup_pairs.append((rank, shipper.id, order))

        if use_flow:
            flow_candidates = [
                (shipper_id, order.id, -rank[1])
                for rank, shipper_id, order in pickup_pairs
            ]
            targets = self._apply_flow_pickups(targets, orders, flow_candidates)
        else:
            used_shippers = set(targets)
            used_orders: set[int] = set()
            for _, shipper_id, order in sorted(pickup_pairs):
                if shipper_id in used_shippers or order.id in used_orders:
                    continue
                targets[shipper_id] = Target("pickup", order.id, (order.sx, order.sy))
                used_shippers.add(shipper_id)
                used_orders.add(order.id)

        return self._apply_sticky_pickups(targets, shippers, orders, t)

    def _assign_targets(self, obs: dict) -> Dict[int, Target]:
        """
        Assign targets in small/medium mode.

        Uses delivery-first assignment, then adaptive flow if the instance is
        difficult, otherwise greedy pickup assignment with beam lookahead.
        """
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

        active_orders = [order for order in orders.values() if not order.picked and not order.delivered]
        if self._should_use_flow_assignment(len(active_orders)):
            flow_candidates: List[Tuple[int, int, float]] = []
            for shipper in shippers:
                if shipper.id in targets:
                    continue
                if len(shipper.bag) >= max(1, shipper.K_max):
                    continue
                for order in active_orders:
                    if not self._can_pickup(shipper, order, orders):
                        continue
                    value = self._pickup_flow_value(shipper, order, orders, t)
                    if value > 0.0:
                        flow_candidates.append((shipper.id, order.id, value))
            return self._apply_flow_pickups(targets, orders, flow_candidates)

        pickup_pairs: List[Tuple[Tuple[float, ...], int, Order]] = []
        for shipper in shippers:
            if shipper.id in targets and len(shipper.bag) >= max(1, shipper.K_max):
                continue
            for order in active_orders:
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
        """Split CBS constraints into vertex[time] and edge[time] lookup tables."""
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
        """Return True if pos->nxt violates a vertex or edge constraint."""
        return nxt in vertex_constraints[depart_t + 1] or (pos, nxt) in edge_constraints[depart_t]

    def _low_level_search(
        self,
        shipper: Shipper,
        target: Target,
        constraints: Tuple[Constraint, ...],
        max_time: int,
    ) -> Optional[List[Position]]:
        """
        Constrained A* for one shipper inside CBS.

        State is (position, time). Cost:
          g = path length so far
          h = cached shortest-path distance to target
          f = g + h
        The search rejects moves violating CBS constraints or unable to reach
        target before max_time.
        """
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
        """Return path[t], or the final cell if the path has already arrived."""
        return path[t] if t < len(path) else path[-1]

    def _first_conflict(
        self,
        paths: Dict[int, List[Position]],
    ) -> Optional[Tuple[str, int, int, Position, Optional[Position], int]]:
        """
        Find the first CBS conflict among paths.

        Vertex conflict: two shippers occupy the same cell at t.
        Edge conflict: two shippers swap cells between t and t+1.
        """
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
        """Push a CBS node ordered by sum-of-costs, conflict flag, then FIFO counter."""
        self._counter += 1
        cost = sum(len(path) - 1 for path in paths.values())
        conflicts_left = 1 if self._first_conflict(paths) else 0
        heapq.heappush(heap, (cost, conflicts_left, self._counter, constraints, paths))

    def _cbs_plan(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Target],
    ) -> Optional[Dict[int, List[Position]]]:
        """
        High-level Conflict-Based Search for small/medium mode.

        Root node plans all target paths with unconstrained A*. When a conflict
        appears, branch by adding one constraint to one conflicting shipper and
        replan only that shipper. The expansion budget is _max_cbs_nodes.
        """
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
        """Convert a planned path to one env action; pickup/deliver only on target arrival."""
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
        """Greedy one-step fallback when CBS cannot find a conflict-free plan."""
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
        """
        Fast large-mode movement toward target.

        Uses cached shortest-path first move, then locally reroutes if the next
        cell is already reserved by a higher-priority shipper.
        """
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
        """
        Decide all actions in large mode without CBS.

        Shippers are ordered by task urgency: deliveries first, then pickups,
        both sorted by deadline/priority. A reserved-cell set avoids simple
        same-cell conflicts cheaply.
        """
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
        """
        Main per-timestep policy.

        Refresh observation-derived state, update hotspot/surge inference, then
        choose the large fast planner or small/medium CBS planner.
        """
        self._refresh_from_obs(obs)
        self._hotspot_tracker.update(obs)
        self._update_surge_score(obs)
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
        """Run online rolling-horizon planning until env.done and return final result."""
        start_time = time.time()
        obs = self.env.reset()
        self._hotspot_tracker.reset()
        self._refresh_from_obs(obs)
        self._new_order_history.clear()
        self._surge_score = 0.0
        self._active_pressure = 0.0
        self._sticky_pickups.clear()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
