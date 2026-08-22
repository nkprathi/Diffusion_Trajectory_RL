"""
plot_diffusion_multipath.py  —  Phase 2 Visualisation

Shows how diffusion generates DIFFERENT paths from the SAME start → SAME goal.
Demonstrates the multi-modal / stochastic nature of the diffusion model.

Outputs:
    results/diffusion_multipath_overlay.png   — all 8 paths overlaid on one grid
    results/diffusion_multipath_panels.png    — 8 individual panels, same start/goal

Usage:
    python phase2_diffusion/visualisation/plot_diffusion_multipath.py
    python phase2_diffusion/visualisation/plot_diffusion_multipath.py --start 1 5 --n 8
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from env.grid_viz import (
    TILE_COLORS,
    STATIC_LEGEND_TILE_ORDER,
    draw_home_grid,
    row_col_center,
)
from env.map_config import GRID_SIZE, SPECIAL_TILES, TILE_LABELS
from phase2_diffusion.noise_schedule import NoiseSchedule
from phase2_diffusion.unet import TemporalUNet
from phase2_diffusion.diffusion import GaussianDiffusion
from phase2_diffusion.trajectory_utils import (
    make_context,
    make_task_context,
    normalize_grid,
    goal_normalized,
    nearest_laundry_cell,
    TASK_CONTEXT_DIM,
)

GRID_SIZE = 12
GOAL_POS  = tuple(SPECIAL_TILES["G"][0])

PALETTE = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]

def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    sched = NoiseSchedule(T=cfg["T"], schedule=cfg["schedule"], device=device)
    unet  = TemporalUNet(
        state_dim=cfg["state_dim"],
        horizon=cfg["horizon"],
        channels=tuple(cfg["channels"]),
        time_embed_dim=cfg["time_embed_dim"],
        context_dim=cfg.get("context_dim", 0),
    )
    unet.load_state_dict(ckpt["model_state"])
    unet.eval()
    return GaussianDiffusion(unet, sched, device=device), cfg


def _denorm(coords: np.ndarray) -> np.ndarray:
    return (coords + 1.0) / 2.0 * (GRID_SIZE - 1)


DEFAULT_START = tuple(SPECIAL_TILES["S"][0])  # (0, 0) — masked in multipath viz
MULTIPATH_MASK_CELLS = {DEFAULT_START}

def _draw_grid(ax, draw_map=True, show_labels=True):
    draw_home_grid(
        ax,
        show_legend=False,
        show_axis_labels=show_labels,
        mask_cells=MULTIPATH_MASK_CELLS,
    )


def _plot_path(ax, traj_grid, color, lw=1.8, alpha=1.0, label=None):
    cols = [row_col_center(int(r), int(c))[0] for r, c in traj_grid]
    rows = [row_col_center(int(r), int(c))[1] for r, c in traj_grid]
    ax.plot(cols, rows, "-", color=color, linewidth=lw, alpha=alpha, label=label, zorder=3)
    ax.scatter(cols[1:-1], rows[1:-1], s=12, color=color, alpha=alpha, zorder=4)


def _mark_endpoints(ax, start_grid, goal_grid, ms=90):
    sx, sy = row_col_center(int(start_grid[0]), int(start_grid[1]))
    gx, gy = row_col_center(int(goal_grid[0]), int(goal_grid[1]))
    ax.scatter(sx, sy, s=ms, color="#2ECC71", marker="^",
               edgecolors="white", linewidth=1.2, zorder=7, label="Start")
    ax.scatter(gx, gy, s=ms, color=TILE_COLORS["G"], marker="*",
               edgecolors="white", linewidth=1.2, zorder=7, label="Washing machine")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Overlay: all 8 paths on ONE grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_overlay(trajs_grid: np.ndarray, start_grid, goal_grid, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    _draw_grid(ax, draw_map=True)

    for i, traj in enumerate(trajs_grid):
        _plot_path(ax, traj, PALETTE[i], lw=2.2, alpha=0.85,
                   label=f"Sample {i+1}")

    _mark_endpoints(ax, start_grid, goal_grid, ms=130)

    legend_tiles = [
        mpatches.Patch(facecolor=TILE_COLORS[k], alpha=0.85, label=TILE_LABELS[k])
        for k in STATIC_LEGEND_TILE_ORDER if k not in ("S", "E")
    ]
    path_handles = [
        plt.Line2D([0], [0], color=PALETTE[i], lw=2, label=f"Sample {i+1}")
        for i in range(len(trajs_grid))
    ]
    # Place legend outside the grid to the right
    ax.legend(handles=legend_tiles + path_handles,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, fontsize=8, framealpha=0.95,
              ncol=1, markerscale=1.0)

    ax.set_title(
        f"Diffusion: 8 Different Paths — Same Start → Same Goal\n"
        f"Start = ({int(start_grid[0])}, {int(start_grid[1])})   Goal = {GOAL_POS}",
        fontsize=11, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=(ax.get_legend(),))
    plt.close(fig)
    print(f"[vis] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Panels: 8 individual grids, same start/goal highlighted
# ─────────────────────────────────────────────────────────────────────────────

def plot_panels(trajs_grid: np.ndarray, start_grid, goal_grid, out_path: str):
    n = len(trajs_grid)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).flatten()

    for i, traj in enumerate(trajs_grid):
        ax = axes[i]
        _draw_grid(ax, show_labels=False)
        _plot_path(ax, traj, PALETTE[i], lw=2.2, alpha=1.0)
        _mark_endpoints(ax, start_grid, goal_grid, ms=80)
        ax.set_title(f"Sample {i+1}", fontsize=10, fontweight="bold",
                     color=PALETTE[i], pad=5)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Diffusion Generates Different Paths — Same Start ({int(start_grid[0])}, {int(start_grid[1])}) → Same Goal {GOAL_POS}\n"
        "Each sample uses a different random noise seed → different trajectory",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[vis] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    start_pos=(0, 3),
    n_samples=8,
    device="cpu",
    temperature=1.2,
    ckpt_name: str = "diffusion_best.pt",
    out_prefix: str | None = None,
):
    out_prefix = out_prefix or (
        "diffusion_taskcond" if ckpt_name == "diffusion_taskcond.pt" else "diffusion"
    )
    ckpt_path   = os.path.join(_ROOT, "checkpoints", ckpt_name)
    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isfile(ckpt_path):
        sys.exit(f"[ERROR] Missing checkpoint: {ckpt_path}")

    diffusion, cfg = _load_model(ckpt_path, device)
    H, D = cfg["horizon"], cfg["state_dim"]
    context_dim = cfg.get("context_dim", 0)

    # Fixed start and goal — same for ALL samples
    start_grid = np.array([int(x) for x in start_pos], dtype=np.float32)
    goal_grid  = np.array(GOAL_POS,  dtype=np.float32)
    start_norm = normalize_grid(start_grid)
    goal_norm  = goal_normalized()

    start_1 = torch.tensor(start_norm, device=device, dtype=torch.float32).unsqueeze(0)
    goal_1  = torch.tensor(goal_norm,  device=device, dtype=torch.float32).unsqueeze(0)

    if context_dim >= TASK_CONTEXT_DIM:
        laundry = nearest_laundry_cell([int(start_grid[0]), int(start_grid[1])])
        laundry_t = torch.tensor(
            normalize_grid(laundry), device=device, dtype=torch.float32,
        ).unsqueeze(0)
        context_1 = make_task_context(start_1, goal_1, laundry_t)
    elif context_dim > 0:
        context_1 = make_context(start_1, goal_1)
    else:
        context_1 = None

    # Generate each sample independently with a different seed for maximum diversity
    print(f"[vis] Generating {n_samples} paths from {tuple(start_pos)} → {GOAL_POS} "
          f"(ckpt={ckpt_name}, temperature={temperature}) …")
    samples_list = []
    for i in range(n_samples):
        torch.manual_seed(i * 7919 + 42)   # reproducible but distinct noise per sample
        sample_kw = dict(
            batch_size=1,
            horizon=H,
            state_dim=D,
            start=start_1,
            goal=goal_1,
            temperature=temperature,
        )
        if context_1 is not None:
            sample_kw["context"] = context_1
        with torch.no_grad():
            s = diffusion.sample(**sample_kw)
        samples_list.append(s)

    samples    = torch.cat(samples_list, dim=0)
    trajs_grid = _denorm(samples.cpu().numpy())   # (N, H, 2)

    # Force exact start/goal on all trajectories
    trajs_grid[:, 0]  = start_grid
    trajs_grid[:, -1] = goal_grid

    plot_overlay(
        trajs_grid, start_grid, goal_grid,
        os.path.join(results_dir, f"{out_prefix}_multipath_overlay.png"),
    )
    plot_panels(
        trajs_grid, start_grid, goal_grid,
        os.path.join(results_dir, f"{out_prefix}_multipath_panels.png"),
    )
    print("[vis] Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start",       type=int,   nargs=2, default=[0, 3],
                   metavar=("ROW", "COL"), help="Start position on grid (default: 0 3)")
    p.add_argument("--n",           type=int,   default=8,   help="Number of samples")
    p.add_argument("--temperature", type=float, default=1.2, help="Initial noise scale for diversity (default: 1.2)")
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--ckpt",        type=str,   default="diffusion_best.pt")
    p.add_argument("--out_prefix",  type=str,   default=None)
    args = p.parse_args()
    run(
        start_pos=tuple(args.start),
        n_samples=args.n,
        device=args.device,
        temperature=args.temperature,
        ckpt_name=args.ckpt,
        out_prefix=args.out_prefix,
    )
