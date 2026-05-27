from __future__ import annotations

from typing import Any


def format_map(obs: dict[str, Any]) -> str:
    """
    Tạo text mô tả trạng thái map hiện tại.

    Ký hiệu:
      #  vật cản
      .  ô trống
      P  điểm lấy hàng của đơn chưa nhặt
      D  điểm giao hàng của đơn chưa giao
      0..9 / A..Z  shipper id
    """
    grid = obs["grid"]
    orders = obs["orders"]
    shippers = obs["shippers"]

    board = [["#" if cell == 1 else "." for cell in row] for row in grid]

    for order in orders.values():
        if not order.delivered:
            board[order.ex][order.ey] = "D"
        if not order.picked:
            board[order.sx][order.sy] = "P"

    for shipper in shippers:
        r, c = shipper.position
        if shipper.id < 10:
            marker = str(shipper.id)
        else:
            marker = chr(ord("A") + (shipper.id - 10) % 26)
        board[r][c] = marker

    active_orders = len(orders)
    carried = sum(len(shipper.bag) for shipper in shippers)
    lines = [
        f"--- MAP t={obs['t']}/{obs['T']} active={active_orders} carried={carried} ---"
    ]
    lines.extend(" ".join(row) for row in board)
    return "\n".join(lines)


def print_map(obs: dict[str, Any]) -> None:
    """In trạng thái map hiện tại ra terminal."""
    print("\n" + format_map(obs))


def map_log_path(cfg: dict[str, Any], prefix: str = "map") -> str:
    """Tên file log map riêng cho từng config."""
    config_name = str(cfg.get("name", "unknown"))
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in config_name)
    return f"{prefix}_{safe_name}.txt"
