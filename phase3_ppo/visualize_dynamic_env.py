"""
visualize_dynamic_env.py — snapshot of dynamic PET obstacle behaviour.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from env import EscapeGameGridEnv, SPECIAL_TILES
from env.dynamic_obstacle import cells_within_radius
from env.grid_viz import draw_home_grid, row_col_center
from env.map_config import DYNAMIC_PET_RADIUS, GRID_SIZE


def simulate_episode(*, seed: int = 0, n_steps: int = 12):
    env = EscapeGameGridEnv(
        grid_size=GRID_SIZE,
        special_tiles=SPECIAL_TILES,
        random_initialization=False,
        headless=True,
        dynamic_pet=True,
        pet_radius=DYNAMIC_PET_RADIUS,
        pet_seed=seed,
    )
    env.reset(seed=seed)
    path = [tuple(env.state)]
    pet_trace = [env.pet_pos]

    for _ in range(n_steps):
        env.step(1)
        path.append(tuple(env.state))
        pet_trace.append(env.pet_pos)

    return env, path, pet_trace


def plot_dynamic_env(out_path: str, *, seed: int = 0, n_steps: int = 12) -> None:
    env, path, pet_trace = simulate_episode(seed=seed, n_steps=n_steps)
    robot = tuple(env.state)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.patch.set_facecolor("white")

    ax0 = axes[0]
    draw_home_grid(
        ax0,
        title=f"Dynamic pet env  (radius r={DYNAMIC_PET_RADIUS} around robot)",
        show_legend=False,
        show_axis_labels=True,
    )

    zone = cells_within_radius(robot, DYNAMIC_PET_RADIUS, GRID_SIZE)
    for r, c in zone:
        ax0.add_patch(
            mpatches.Rectangle(
                (c, r), 1, 1,
                facecolor="#E8DAEF", edgecolor="none", alpha=0.45, zorder=2,
            )
        )

    rx, ry = row_col_center(*robot)
    ax0.plot(rx, ry, "o", color="#0066FF", markersize=14, zorder=7)

    if env.pet_pos is not None:
        px, py = row_col_center(*env.pet_pos)
        ax0.plot(px, py, "s", color="#8E44AD", markersize=15, zorder=8)
        ax0.text(px, py - 0.35, "PET", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#6C3483")

    legend_items = [
        ("#0066FF", "Robot"),
        ("#8E44AD", "PET (dynamic)"),
        ("#E8DAEF", f"Near-robot zone (r={DYNAMIC_PET_RADIUS})"),
    ]
    handles = [
        mpatches.Patch(facecolor=c, edgecolor="#888", label=label)
        for c, label in legend_items
    ]
    ax0.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    ax1 = axes[1]
    draw_home_grid(ax1, title="Robot path (random walk)", show_legend=False)
    xs = [row_col_center(r, c)[0] for r, c in path]
    ys = [row_col_center(r, c)[1] for r, c in path]
    ax1.plot(xs, ys, "-o", color="#2980B9", markersize=4, linewidth=1.5, zorder=3)

    valid_pet = [p for p in pet_trace if p is not None]
    if valid_pet:
        pxs = [row_col_center(r, c)[0] for r, c in valid_pet]
        pys = [row_col_center(r, c)[1] for r, c in valid_pet]
        ax1.plot(pxs, pys, "s--", color="#8E44AD", markersize=6, linewidth=1.2,
                 alpha=0.7, label="PET trace", zorder=4)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=12)
    p.add_argument("--out", type=str,
                   default=os.path.join(_ROOT, "results", "dynamic_pet_env.png"))
    args = p.parse_args()
    plot_dynamic_env(args.out, seed=args.seed, n_steps=args.n_steps)


if __name__ == "__main__":
    main()
