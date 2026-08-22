"""
home_route_planner.py — Grid A* routing for the home navigation task.

Plans: robot start → laundry bag pickup(s) → washing machine (goal).
Blocked for planning: furniture (W), staircase (F).
Dynamic pet obstacle (PET) is handled in Phase 3 only.
"""

from __future__ import annotations

import heapq
import itertools
from typing import List, Optional, Sequence, Set, Tuple

GridCell = Tuple[int, int]

# Tiles treated as impassable for geometric planning (static layout only)
BLOCKED_TILES = frozenset({"W", "F"})


def build_blocked_set(
    special_tiles: dict,
    grid_size: int = 12,
) -> Set[GridCell]:
    """Return set of (row, col) cells that A* cannot enter."""
    blocked: Set[GridCell] = set()
    for tile in BLOCKED_TILES:
        for coord in special_tiles.get(tile, []):
            r, c = int(coord[0]), int(coord[1])
            if 0 <= r < grid_size and 0 <= c < grid_size:
                blocked.add((r, c))
    return blocked


def list_empty_cells(
    special_tiles: dict,
    grid_size: int = 12,
    blocked: Optional[Set[GridCell]] = None,
) -> List[GridCell]:
    """Cells with no special tile label (walkable spawn points)."""
    if blocked is None:
        blocked = build_blocked_set(special_tiles, grid_size)

    occupied: Set[GridCell] = set()
    for coords in special_tiles.values():
        for coord in coords:
            occupied.add((int(coord[0]), int(coord[1])))

    empties: List[GridCell] = []
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) in blocked:
                continue
            if (r, c) not in occupied:
                empties.append((r, c))
    return empties


def _in_bounds(cell: GridCell, grid_size: int) -> bool:
    r, c = cell
    return 0 <= r < grid_size and 0 <= c < grid_size


def _neighbors(cell: GridCell, grid_size: int, blocked: Set[GridCell]) -> List[GridCell]:
    r, c = cell
    out: List[GridCell] = []
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        nr, nc = r + dr, c + dc
        nxt = (nr, nc)
        if _in_bounds(nxt, grid_size) and nxt not in blocked:
            out.append(nxt)
    return out


def _manhattan(a: GridCell, b: GridCell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    start: GridCell,
    goal: GridCell,
    blocked: Set[GridCell],
    grid_size: int = 12,
) -> Optional[List[GridCell]]:
    """
    4-connected A* with Manhattan heuristic.
    Returns list of (row, col) from start to goal inclusive, or None if unreachable.
    """
    if start in blocked or goal in blocked:
        return None
    if start == goal:
        return [start]

    open_heap: List[Tuple[int, int, GridCell]] = []
    counter = 0
    g_score = {start: 0}
    came_from: dict[GridCell, GridCell] = {}

    f0 = _manhattan(start, goal)
    heapq.heappush(open_heap, (f0, counter, start))
    counter += 1

    closed: Set[GridCell] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)

        for nxt in _neighbors(current, grid_size, blocked):
            tentative = g_score[current] + 1
            if tentative < g_score.get(nxt, 10**9):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + _manhattan(nxt, goal)
                heapq.heappush(open_heap, (f, counter, nxt))
                counter += 1

    return None


def _concat_paths(segments: Sequence[List[GridCell]]) -> List[GridCell]:
    if not segments:
        return []
    out = list(segments[0])
    for seg in segments[1:]:
        out.extend(seg[1:])
    return out


def plan_through_waypoints(
    start: GridCell,
    waypoints: Sequence[GridCell],
    goal: GridCell,
    blocked: Set[GridCell],
    grid_size: int = 12,
) -> Optional[List[GridCell]]:
    """Plan start → w1 → … → wk → goal."""
    full: List[GridCell] = []
    current = start
    for wp in list(waypoints) + [goal]:
        segment = astar(current, wp, blocked, grid_size)
        if segment is None:
            return None
        if full:
            full.extend(segment[1:])
        else:
            full.extend(segment)
        current = wp
    return full


def greedy_coin_order(
    start: GridCell,
    coins: Sequence[GridCell],
    n_coins: int,
    blocked: Set[GridCell],
    grid_size: int = 12,
) -> Optional[List[GridCell]]:
    """
    Repeatedly visit the nearest reachable remaining coin.
    Returns ordered list of n_coins coin cells.
    """
    remaining = list(coins)
    order: List[GridCell] = []
    current = start

    for _ in range(n_coins):
        best: Optional[GridCell] = None
        best_len: Optional[int] = None
        for coin in remaining:
            seg = astar(current, coin, blocked, grid_size)
            if seg is None:
                continue
            length = len(seg) - 1
            if best_len is None or length < best_len:
                best_len = length
                best = coin
        if best is None:
            return None
        order.append(best)
        remaining.remove(best)
        current = best
    return order


def best_coin_order_exhaustive(
    start: GridCell,
    coins: Sequence[GridCell],
    n_coins: int,
    goal: GridCell,
    blocked: Set[GridCell],
    grid_size: int = 12,
    max_permutations: int = 5000,
) -> Optional[Tuple[List[GridCell], List[GridCell]]]:
    """
    Try combinations/permutations of n_coins from coins; pick shortest full path to goal.
    Returns (coin_order, full_path) or None.
    """
    if len(coins) < n_coins:
        return None

    best_path: Optional[List[GridCell]] = None
    best_order: Optional[List[GridCell]] = None
    checked = 0

    for combo in itertools.combinations(coins, n_coins):
        for perm in itertools.permutations(combo):
            checked += 1
            if checked > max_permutations:
                break
            path = plan_through_waypoints(start, perm, goal, blocked, grid_size)
            if path is None:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_order = list(perm)
        if checked > max_permutations:
            break

    if best_path is None or best_order is None:
        return None
    return best_order, best_path


def plan_home_route(
    start: GridCell,
    goal: GridCell,
    laundry_cells: Sequence[GridCell],
    blocked: Set[GridCell],
    min_laundry: int = 1,
    grid_size: int = 12,
    *,
    optimize_order: bool = True,
) -> Optional[dict]:
    """
    Build a path: start → (≥ min_laundry piles) → washing machine (goal).

    optimize_order=True: pick best laundry pile / order (shortest path).
    optimize_order=False: greedy nearest laundry (faster).
    """
    if start in blocked or goal in blocked:
        return None

    piles = [tuple(map(int, c)) for c in laundry_cells]

    if optimize_order and len(piles) >= min_laundry:
        result = best_coin_order_exhaustive(
            start, piles, min_laundry, goal, blocked, grid_size
        )
        if result is not None:
            order, path = result
            return {
                "positions": path,
                "laundry_collected": len(order),
                "coins_collected": len(order),
                "steps": len(path) - 1,
                "total_reward": None,
                "pickup_order": order,
                "coin_order": order,
            }

    order = greedy_coin_order(start, piles, min_laundry, blocked, grid_size)
    if order is None:
        return None
    path = plan_through_waypoints(start, order, goal, blocked, grid_size)
    if path is None:
        return None

    return {
        "positions": path,
        "laundry_collected": len(order),
        "coins_collected": len(order),
        "steps": len(path) - 1,
        "total_reward": None,
        "pickup_order": order,
        "coin_order": order,
    }


# Backward-compatible alias
plan_multi_goal_with_coins = plan_home_route
