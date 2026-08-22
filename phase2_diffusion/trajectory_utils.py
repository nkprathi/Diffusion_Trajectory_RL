"""
trajectory_utils.py — shared normalization and goal-conditioning helpers.

Context variants:
    CONTEXT_DIM (4)      — [start; goal]  (Phase 2 baseline)
    TASK_CONTEXT_DIM (6) — [start; goal; laundry_target]  (Phase 2 diffusion)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

GRID_SIZE = 12
CONTEXT_DIM = 4
TASK_CONTEXT_DIM = 6
GOAL_GRID = (10, 10)
LAUNDRY_CELLS = [(1, 11), (7, 3)]


def normalize_grid(coords: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Map grid coords [0, grid_size-1] → [-1, 1]."""
    return (2.0 * np.asarray(coords, dtype=np.float32) / (grid_size - 1) - 1.0)


def goal_normalized(grid_size: int = GRID_SIZE) -> np.ndarray:
    """Normalized goal position for the escape game."""
    return normalize_grid(np.array(GOAL_GRID, dtype=np.float32), grid_size)


def make_context_from_batch(batch: Tensor) -> Tensor:
    """
    Build 4-dim goal-conditioning from trajectory batch.

    Returns (B, 4) = [start_row, start_col, goal_row, goal_col]
    """
    start = batch[:, 0, :]
    goal = batch[:, -1, :]
    return torch.cat([start, goal], dim=-1)


def make_context(
    start_norm: Tensor,
    goal_norm: Tensor,
) -> Tensor:
    """(B, 4) context from start and goal."""
    if start_norm.dim() == 1:
        start_norm = start_norm.unsqueeze(0)
    if goal_norm.dim() == 1:
        goal_norm = goal_norm.unsqueeze(0)
    return torch.cat([start_norm, goal_norm], dim=-1)


def make_task_context(
    start_norm: Tensor,
    goal_norm: Tensor,
    laundry_norm: Tensor,
) -> Tensor:
    """(B, 6) = [start(2); goal(2); laundry_target(2)]."""
    if start_norm.dim() == 1:
        start_norm = start_norm.unsqueeze(0)
    if goal_norm.dim() == 1:
        goal_norm = goal_norm.unsqueeze(0)
    if laundry_norm.dim() == 1:
        laundry_norm = laundry_norm.unsqueeze(0)
    return torch.cat([start_norm, goal_norm, laundry_norm], dim=-1)


def laundry_cell_from_path(path: np.ndarray, grid_size: int = GRID_SIZE) -> tuple[int, int]:
    """
    First laundry bag visited along a grid path (row, col).
    Falls back to nearest laundry cell to start if none visited.
    """
    laundry_set = {tuple(c) for c in LAUNDRY_CELLS}
    for pt in path:
        r, c = int(round(float(pt[0]))), int(round(float(pt[1])))
        if 0 <= r < grid_size and 0 <= c < grid_size and (r, c) in laundry_set:
            return (r, c)
    start = path[0]
    best, best_d = LAUNDRY_CELLS[0], 1e9
    for cell in laundry_set:
        d = abs(cell[0] - start[0]) + abs(cell[1] - start[1])
        if d < best_d:
            best_d, best = d, cell
    return best


def laundry_cell_from_meta(meta: dict | None, path: np.ndarray) -> tuple[int, int]:
    """Prefer planner pickup order; else infer from path geometry."""
    if meta is not None:
        order = meta.get("pickup_order") or meta.get("laundry_collected")
        if order and len(order) > 0:
            cell = order[0]
            return (int(cell[0]), int(cell[1]))
    return laundry_cell_from_path(path)


def make_task_context_from_batch(
    batch: Tensor,
    laundry_norm: Tensor,
) -> Tensor:
    """Build (B, 6) context from trajectories + precomputed laundry targets."""
    base = make_context_from_batch(batch)
    if laundry_norm.dim() == 1:
        laundry_norm = laundry_norm.unsqueeze(0)
    return torch.cat([base, laundry_norm], dim=-1)


def make_task_context_for_pos(
    start_pos: list | tuple,
    *,
    laundry_cell: tuple[int, int] | None = None,
    goal_grid: tuple[int, int] = GOAL_GRID,
) -> np.ndarray:
    """Numpy (6,) task context for sampling at inference."""
    start_n = normalize_grid(start_pos)
    goal_n = goal_normalized()
    if laundry_cell is None:
        laundry_cell = laundry_cell_from_path(
            np.array([[start_pos[0], start_pos[1]], list(goal_grid)], dtype=np.float32)
        )
    laundry_n = normalize_grid(laundry_cell)
    return np.concatenate([start_n, goal_n, laundry_n], dtype=np.float32)


def nearest_laundry_cell(
    pos: list | tuple,
    *,
    collected_mask: list[bool] | None = None,
) -> tuple[int, int]:
    """Nearest uncollected laundry pile to pos (Manhattan)."""
    collected_mask = collected_mask or [False] * len(LAUNDRY_CELLS)
    best, best_d = LAUNDRY_CELLS[0], 1e9
    for idx, cell in enumerate(LAUNDRY_CELLS):
        if idx < len(collected_mask) and collected_mask[idx]:
            continue
        d = abs(cell[0] - int(pos[0])) + abs(cell[1] - int(pos[1]))
        if d < best_d:
            best_d, best = d, cell
    return best
