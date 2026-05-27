from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple


SEED = 20260526
OUT_PATH = "validation_config.txt"


def free_cells(grid: List[List[int]]) -> List[Tuple[int, int]]:
    return [(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 0]


def is_connected(grid: List[List[int]]) -> bool:
    cells = free_cells(grid)
    if not cells:
        return False

    seen = {cells[0]}
    queue: deque[Tuple[int, int]] = deque([cells[0]])
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if 0 <= nr < len(grid) and 0 <= nc < len(grid) and grid[nr][nc] == 0 and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def make_grid(n: int, obstacle_prob: float, rng: random.Random) -> List[List[int]]:
    for _ in range(500):
        grid = [[1 if r in {0, n - 1} or c in {0, n - 1} else 0 for c in range(n)] for r in range(n)]
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                if rng.random() < obstacle_prob:
                    grid[r][c] = 1

        # Add a few bottlenecks, but always leave gates open.
        if n >= 12:
            mid = n // 2
            for c in range(1, n - 1):
                if c not in {mid - 1, mid, mid + 1} and rng.random() < 0.55:
                    grid[mid][c] = 1
            for r in range(1, n - 1):
                if r not in {mid - 1, mid, mid + 1} and rng.random() < 0.45:
                    grid[r][mid] = 1

        if is_connected(grid):
            return grid

    raise RuntimeError(f"Could not generate connected grid for N={n}")


def config_block(name: str, n: int, c: int, g: int, t: int, obstacle_prob: float, rng: random.Random) -> str:
    grid = make_grid(n, obstacle_prob, rng)
    k_values = [rng.choice([2, 3]) for _ in range(c)]
    w_values = [rng.choice([20.0, 30.0]) for _ in range(c)]
    rows = [
        "[CONFIG]",
        f"name    = {name}",
        f"N       = {n}",
        f"C       = {c}",
        f"G       = {g}",
        f"T       = {t}",
        "K_max   = " + " ".join(str(v) for v in k_values),
        "W_max   = " + " ".join(f"{v:.1f}" for v in w_values),
        "[MAP]",
    ]
    rows.extend(" ".join(str(v) for v in row) for row in grid)
    rows.append("[END]")
    return "\n".join(rows)


def main() -> None:
    rng = random.Random(SEED)
    specs = [
        ("V1", 8, 2, 18, 220, 0.08),
        ("V2", 11, 3, 32, 300, 0.10),
        ("V3", 13, 3, 45, 380, 0.12),
        ("V4", 16, 4, 65, 520, 0.13),
        ("V5", 18, 5, 85, 620, 0.11),
        ("V6", 20, 5, 110, 760, 0.12),
        ("V7", 14, 4, 55, 440, 0.18),
        ("V8", 19, 5, 95, 700, 0.17),
    ]
    content = "\n\n".join(config_block(*spec, rng) for spec in specs)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(content + "\n")


if __name__ == "__main__":
    main()
