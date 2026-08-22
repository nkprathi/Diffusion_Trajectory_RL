"""
plot_method_comparison.py — Side-by-side grid map for the same start → goal.

Compares executed paths:
    A* open-loop | Diffusion waypoints | PPO

Output: results/eval_method_comparison.png (generated on each evaluate() run)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env.grid_viz import (
    draw_home_grid,
    plot_goal_marker,
    plot_grid_path,
    plot_start_marker,
    plot_waypoint_plan,
)
from env.map_config import GRID_SIZE, SPECIAL_TILES
from phase2_diffusion.trajectory_utils import nearest_laundry_cell

DEFAULT_COMPARISON_START = (0, 3)
DEFAULT_GOAL = tuple(SPECIAL_TILES["G"][0])
MASK_START_CELL = {tuple(SPECIAL_TILES["S"][0])}

COMPARISON_METHODS = [
    "A* open-loop",
    "Diffusion waypoints",
    "PPO",
]

METHOD_COLORS = {
    "A* open-loop": "#4C72B0",
    "Diffusion waypoints": "#C44E52",
    "PPO": "#DD8452",
}
METHOD_TITLE_COLOR = "#2C3E50"

DEFAULT_COMPARISON_PET_SEED = 3


def comparison_mask_cells() -> set[tuple[int, int]]:
    """Hide only the default S tile; show both laundry bags on the map."""
    return set(MASK_START_CELL)


def comparison_laundry_cells(start: tuple[int, int]) -> list[tuple[int, int]]:
    """Nearest laundry target for fixed-start comparison episodes."""
    return [nearest_laundry_cell(start)]


def _greedy_action_toward(state, target) -> int:
    r0, c0 = int(state[0]), int(state[1])
    r1, c1 = int(target[0]), int(target[1])
    dr, dc = r1 - r0, c1 - c0
    if abs(dr) >= abs(dc) and dr != 0:
        return 0 if dr < 0 else 1
    if dc != 0:
        return 2 if dc > 0 else 3
    return 0 if dr < 0 else 1


def _run_comparison_ppo_episode(
    env,
    model,
    *,
    max_steps: int,
    is_task_success,
) -> dict:
    """
    Comparison-only PPO episode: follow diffusion waypoints until nearest
    laundry (7, 3) is collected, then run the trained PPO policy for delivery.
    """
    base_obs = np.array(
        env.base_env.state + [env.base_env._coin_mask()], dtype=np.int32,
    )
    obs = env._make_obs(base_obs)
    total_r, steps = 0.0, 0
    path = [env.base_env.state[:]]
    done = False
    info: dict = {}

    while not done and steps < max_steps:
        if not env._laundry_satisfied():
            wp = env._waypoints[env._wp_idx]
            action = _greedy_action_toward(env.base_env.state, wp)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, r, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_r += r
        steps += 1
        path.append(env.base_env.state[:])

    return {
        "reward": total_r,
        "length": steps,
        "success": 1 if is_task_success(env.base_env, total_reward=total_r) else 0,
        "path": np.array(path),
        "waypoints_done": info.get("waypoints_done", 0),
    }


def pick_comparison_pet_seed(
    *,
    start: tuple[int, int] = DEFAULT_COMPARISON_START,
    device: str = "cpu",
    ppo_ckpt: str | None = None,
    diffusion_ckpt: str | None = None,
    seed_range: int = 48,
) -> int:
    """Pick pet_seed where A* and comparison PPO succeed."""
    checkpoint_dir = os.path.join(_ROOT, "checkpoints")
    from phase2_diffusion.checkpoint_names import DIFFUSION_BEST, resolve_checkpoint

    diffusion_ckpt = diffusion_ckpt or resolve_checkpoint(checkpoint_dir, DIFFUSION_BEST)
    ppo_ckpt = ppo_ckpt or os.path.join(checkpoint_dir, "ppo_best.zip")
    if not os.path.isfile(diffusion_ckpt) or not os.path.isfile(ppo_ckpt):
        return DEFAULT_COMPARISON_PET_SEED

    best_seed = DEFAULT_COMPARISON_PET_SEED
    best_score = -1

    for pet_seed in range(seed_range):
        episodes = run_fixed_comparison_episodes(
            start=start,
            pet_seed=pet_seed,
            device=device,
            ppo_ckpt=ppo_ckpt,
            diffusion_ckpt=diffusion_ckpt,
            auto_pick_seed=False,
        )
        astar = episodes.get("A* open-loop", {})
        ppo = episodes.get("PPO", {})
        if not (astar.get("success") and ppo.get("success")):
            continue
        score = 20 + max(0, 50 - int(ppo.get("length", 99)))
        if score > best_score:
            best_score = score
            best_seed = pet_seed

    return best_seed


def run_fixed_comparison_episodes(
    *,
    start: tuple[int, int] = DEFAULT_COMPARISON_START,
    pet_seed: int = DEFAULT_COMPARISON_PET_SEED,
    device: str = "cpu",
    ppo_ckpt: str | None = None,
    diffusion_ckpt: str | None = None,
    periodic_replan: int | None = None,
    min_laundry: int | None = None,
    auto_pick_seed: bool = True,
) -> dict[str, dict]:
    """Run one episode per method from the same fixed start."""
    from env import EscapeGameGridEnv, SPECIAL_TILES as ENV_TILES
    from env.map_config import MIN_LAUNDRY_REQUIRED
    from phase1_home_navigation.home_route_planner import build_blocked_set
    from phase2_diffusion.checkpoint_names import DIFFUSION_BEST, resolve_checkpoint
    from phase3_ppo.env_wrapper import DiffusionWaypointEnv
    from phase3_ppo.evaluate import (
        MAX_EVAL_STEPS,
        _run_astar_episode,
        _run_diffusion_open_loop_episode,
    )
    from phase3_ppo.success_metrics import is_task_success

    if min_laundry is None:
        min_laundry = MIN_LAUNDRY_REQUIRED

    if auto_pick_seed:
        pet_seed = pick_comparison_pet_seed(
            start=start,
            device=device,
            ppo_ckpt=ppo_ckpt,
            diffusion_ckpt=diffusion_ckpt,
        )

    checkpoint_dir = os.path.join(_ROOT, "checkpoints")
    diffusion_ckpt = diffusion_ckpt or resolve_checkpoint(checkpoint_dir, DIFFUSION_BEST)
    ppo_ckpt = ppo_ckpt or os.path.join(checkpoint_dir, "ppo_best.zip")

    blocked = build_blocked_set(ENV_TILES, GRID_SIZE)
    laundry_target = comparison_laundry_cells(start)
    episodes: dict[str, dict] = {}

    reset_opts = {
        "fixed_start": start,
        "pet_seed": pet_seed,
        "laundry_cells": laundry_target,
    }

    # --- A* open-loop (nearest laundry) ---
    env = EscapeGameGridEnv(
        grid_size=GRID_SIZE,
        special_tiles=ENV_TILES,
        random_initialization=False,
        headless=True,
        dynamic_pet=True,
        pet_seed=pet_seed,
    )
    env.reset(seed=pet_seed, options=reset_opts)
    episodes["A* open-loop"] = _run_astar_episode(
        env,
        blocked,
        enable_replan=False,
        min_laundry=min_laundry,
        skip_reset=True,
        laundry_cells=laundry_target,
    )
    env.close()

    # --- Diffusion waypoints (open-loop) ---
    if os.path.isfile(diffusion_ckpt):
        import torch

        torch.manual_seed(pet_seed * 7919 + 42)
        planner = DiffusionWaypointEnv(
            ckpt_path=diffusion_ckpt,
            device=device,
            random_start=False,
        )
        planner.reset(seed=pet_seed, options=reset_opts)
        ep_ol = _run_diffusion_open_loop_episode(planner, skip_reset=True)
        ep_ol["waypoints"] = planner._waypoints.copy()
        episodes["Diffusion waypoints"] = ep_ol
        planner.close()

    # --- PPO (comparison: collect nearest laundry, then PPO delivery) ---
    if os.path.isfile(diffusion_ckpt) and os.path.isfile(ppo_ckpt):
        import torch
        from stable_baselines3 import PPO as SB3PPO

        torch.manual_seed(pet_seed * 7919 + 42)
        model = SB3PPO.load(ppo_ckpt, device=device)
        env = DiffusionWaypointEnv(
            ckpt_path=diffusion_ckpt,
            device=device,
            random_start=False,
        )
        env.reset(seed=pet_seed, options=reset_opts)
        episodes["PPO"] = _run_comparison_ppo_episode(
            env,
            model,
            max_steps=MAX_EVAL_STEPS,
            is_task_success=is_task_success,
        )
        env.close()

    episodes["_pet_seed"] = {"pet_seed": pet_seed}
    return episodes


def plot_method_comparison_map(
    episodes: dict[str, dict],
    out_path: str,
    *,
    start: tuple[int, int] = DEFAULT_COMPARISON_START,
    goal: tuple[int, int] = DEFAULT_GOAL,
) -> Optional[str]:
    """1×3 home-grid panels — method names only, goal star on G tile."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    names = [m for m in COMPARISON_METHODS if m in episodes]
    if not names:
        return None

    mask_cells = comparison_mask_cells()
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.2, 4.2))
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, name in zip(axes_flat, names):
        ep = episodes[name]
        path = ep["path"]
        if len(path) < 2:
            path = np.array([list(start), list(goal)], dtype=np.float32)

        draw_home_grid(
            ax,
            show_legend=False,
            show_axis_labels=False,
            mask_cells=mask_cells,
        )

        if name == "Diffusion waypoints" and ep.get("waypoints") is not None:
            wps = ep["waypoints"]
            wp_cells = [
                (
                    int(np.clip(round(wps[i, 0]), 0, GRID_SIZE - 1)),
                    int(np.clip(round(wps[i, 1]), 0, GRID_SIZE - 1)),
                )
                for i in range(len(wps))
            ]
            plot_waypoint_plan(ax, wp_cells, plan_color=METHOD_COLORS[name], linewidth=1.6)

        dedup_path = []
        for r, c in path:
            cell = (int(r), int(c))
            if not dedup_path or dedup_path[-1] != cell:
                dedup_path.append(cell)

        plot_grid_path(
            ax,
            dedup_path,
            path_color=METHOD_COLORS.get(name, "#333333"),
            linewidth=2.0,
            show_start_marker=False,
            show_end_marker=False,
        )
        plot_start_marker(ax, start)
        plot_goal_marker(ax, goal)

        ax.set_title(
            name,
            fontsize=10,
            fontweight="bold",
            color=METHOD_TITLE_COLOR,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Method Comparison — Start {start} → Goal {goal}",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] Method comparison map → {out_path}")
    return out_path


def generate_method_comparison(
    results_dir: str | None = None,
    *,
    start: tuple[int, int] = DEFAULT_COMPARISON_START,
    pet_seed: int | None = None,
    device: str = "cpu",
    ppo_ckpt: str | None = None,
    diffusion_ckpt: str | None = None,
    periodic_replan: int | None = None,
    out_name: str = "eval_method_comparison.png",
    success_rates: dict | None = None,
    auto_pick_seed: bool = True,
) -> Optional[str]:
    """Run fixed-start episodes and save comparison figure."""
    results_dir = results_dir or os.path.join(_ROOT, "results")

    episodes = run_fixed_comparison_episodes(
        start=start,
        pet_seed=pet_seed if pet_seed is not None else DEFAULT_COMPARISON_PET_SEED,
        device=device,
        ppo_ckpt=ppo_ckpt,
        diffusion_ckpt=diffusion_ckpt,
        periodic_replan=periodic_replan,
        auto_pick_seed=auto_pick_seed and pet_seed is None,
    )
    episodes = {k: v for k, v in episodes.items() if not k.startswith("_")}

    return plot_method_comparison_map(
        episodes,
        os.path.join(results_dir, out_name),
        start=start,
        goal=DEFAULT_GOAL,
    )


# Backward-compatible helper (eval may still import)
def load_eval_success_rates(results_dir: str | None = None) -> dict[str, float]:
    import csv
    results_dir = results_dir or os.path.join(_ROOT, "results")
    csv_path = os.path.join(results_dir, "eval_summary.csv")
    rates: dict[str, float] = {}
    if not os.path.isfile(csv_path):
        return rates
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("method", "").strip()
            if name:
                try:
                    rates[name] = float(row["success_pct"])
                except (KeyError, ValueError):
                    pass
    return rates
