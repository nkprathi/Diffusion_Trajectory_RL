"""Shared episode success criteria (strict task completion)."""

from __future__ import annotations

from typing import Any


def is_task_success(base_env: Any, *, total_reward: float | None = None) -> bool:
    """
    True only when the robot is on the washing machine (G) tile with
    enough laundry collected — same rule for A*, PPO, and training logs.
    """
    row, col = int(base_env.state[0]), int(base_env.state[1])
    laundry_ok = sum(base_env.coins) >= base_env.min_laundry_required
    on_washer = str(base_env.grid[row, col]) == "G"
    if not (laundry_ok and on_washer):
        return False
    if total_reward is not None and total_reward < 250:
        return False
    return True
