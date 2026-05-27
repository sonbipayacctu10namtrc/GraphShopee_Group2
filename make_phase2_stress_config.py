from __future__ import annotations

import argparse
import random
from collections import deque
from typing import List, Tuple


SEED = 20260527
OUT_PATH = "phase2_stress_config.txt"


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
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < len(grid) and 0 <= nc < len(grid) and grid[nr][nc] == 0 and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def carve_arteries(grid: List[List[int]], step: int) -> None:
    n = len(grid)
    for r in range(1, n - 1, step):
        for c in range(1, n - 1):
            grid[r][c] = 0
    for c in range(1, n - 1, step):
        for r in range(1, n - 1):
            grid[r][c] = 0


def make_grid(n: int, obstacle_prob: float, rng: random.Random) -> List[List[int]]:
    for _ in range(200):
        grid = [[1 if r in {0, n - 1} or c in {0, n - 1} else 0 for c in range(n)] for r in range(n)]
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                if rng.random() < obstacle_prob:
                    grid[r][c] = 1

        carve_arteries(grid, max(8, n // 8))
        if is_connected(grid):
            return grid

    raise RuntimeError(f"Could not generate connected grid for N={n}")


def config_block(name: str, n: int, c: int, g: int, t: int, obstacle_prob: float, rng: random.Random) -> str:
    grid = make_grid(n, obstacle_prob, rng)
    k_values = [rng.choice([2, 3, 4]) for _ in range(c)]
    w_values = [rng.choice([20.0, 30.0, 45.0]) for _ in range(c)]
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
    parser = argparse.ArgumentParser(description="Generate connected Phase 2 stress configs.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    specs = [
        ("P2S1", 25, 6, 180, 600, 0.10),
        ("P2S2", 35, 8, 300, 900, 0.11),
        ("P2S3", 45, 10, 450, 1100, 0.12),
        ("P2S4", 55, 12, 650, 1400, 0.12),
        ("P2S5", 65, 15, 850, 1700, 0.13),
        ("P2S6", 75, 18, 1000, 2000, 0.13),
        ("P2S7", 90, 22, 1300, 2200, 0.14),
        ("P2S8", 100, 25, 1500, 2400, 0.14),
    ]
    content = "\n\n".join(config_block(*spec, rng) for spec in specs)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(content + "\n")


if __name__ == "__main__":
    main()
