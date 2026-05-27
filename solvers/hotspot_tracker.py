from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Tuple


Position = Tuple[int, int]


class HotspotTracker:
    """
    Phát hiện hotspot online từ các đơn mới được reveal trong observation.

    Tracker không đọc tham số ẩn trong env.cfg. Nó chỉ dùng:
    - obs["new_order_ids"] để biết đơn nào vừa xuất hiện;
    - obs["orders"] để lấy vị trí pickup của các đơn đó.

    Ý tưởng: giữ sliding window các pickup gần đây, đếm mật độ trong bán kính
    Manhattan nhỏ, rồi xem các vùng mật độ cao là hotspot tạm thời.
    """

    def __init__(self, window: int = 80, radius: int = 3, max_hotspots: int = 3, decay_tau: float = 80.0):
        self.window = max(1, window)
        self.radius = max(0, radius)
        self.max_hotspots = max(1, max_hotspots)
        self.decay_tau = max(1.0, decay_tau)
        self._recent: Deque[Tuple[int, Position]] = deque()
        self._seen_order_ids: set[int] = set()
        self._hotspots: List[Tuple[Position, float]] = []

    def reset(self) -> None:
        self._recent.clear()
        self._seen_order_ids.clear()
        self._hotspots = []

    def update(self, obs: dict[str, Any]) -> None:
        """Cập nhật tracker bằng các đơn mới ở timestep hiện tại."""
        t = int(obs.get("t", 0))
        orders: Dict[int, Any] = obs.get("orders", {})

        for oid in obs.get("new_order_ids", []):
            if oid in self._seen_order_ids:
                continue
            order = orders.get(oid)
            if order is None:
                continue
            self._seen_order_ids.add(oid)
            self._recent.append((t, (order.sx, order.sy)))

        while self._recent and t - self._recent[0][0] > self.window:
            self._recent.popleft()

        self._recompute_hotspots(t)

    def hotspots(self) -> List[Position]:
        """Trả các tâm hotspot tạm thời, đã sắp theo mật độ giảm dần."""
        return [pos for pos, _ in self._hotspots]

    def score(self, pos: Position) -> float:
        """
        Điểm hotspot tại pos trong [0, 1].

        0 nghĩa là không gần cụm pickup gần đây. 1 nghĩa là pos nằm trong cụm
        mạnh nhất đang quan sát được.
        """
        if not self._hotspots:
            return 0.0

        best_count = max(count for _, count in self._hotspots)
        if best_count <= 0:
            return 0.0

        best_score = 0.0
        for center, count in self._hotspots:
            distance = abs(pos[0] - center[0]) + abs(pos[1] - center[1])
            if distance > self.radius:
                continue
            proximity = 1.0 - distance / max(self.radius + 1, 1)
            best_score = max(best_score, (count / best_count) * proximity)
        return best_score

    def _age_weight(self, current_t: int, order_t: int) -> float:
        return math.exp(-max(0, current_t - order_t) / self.decay_tau)

    def _recompute_hotspots(self, current_t: int) -> None:
        if not self._recent:
            self._hotspots = []
            return

        recent = list(self._recent)
        positions = [pos for _, pos in recent]
        scored: List[Tuple[Position, float]] = []
        for pos in set(positions):
            count = sum(
                self._age_weight(current_t, order_t)
                for order_t, other in recent
                if abs(pos[0] - other[0]) + abs(pos[1] - other[1]) <= self.radius
            )
            scored.append((pos, count))

        scored.sort(key=lambda item: (-item[1], item[0]))
        selected: List[Tuple[Position, int]] = []
        for pos, count in scored:
            if count < 0.8:
                continue
            if any(abs(pos[0] - kept[0][0]) + abs(pos[1] - kept[0][1]) <= self.radius for kept in selected):
                continue
            selected.append((pos, count))
            if len(selected) >= self.max_hotspots:
                break

        self._hotspots = selected
