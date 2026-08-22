"""
visualize_home_grid.py — Phase 1 home-grid figures (expert laundry routes).

Coordinate axes: row and column ticks match env state — cell (5, 6) is
row tick 5, column tick 6 (counting from 0 at top-left).

Outputs:
    results/phase1_laundry_route_example_{row}_{col}.png  — one per laundry bag
    results/sample_expert_routes.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env.grid_viz import draw_home_grid, legend_patches, plot_grid_path
from env.map_config import GRID_SIZE, MIN_LAUNDRY_REQUIRED, SPECIAL_TILES
from phase1_home_navigation.home_route_planner import build_blocked_set, plan_home_route

DEFAULT_GOAL = tuple(SPECIAL_TILES["G"][0])
DEFAULT_LAUNDRY = [tuple(c) for c in SPECIAL_TILES["LB"]]
DEFAULT_START = tuple(SPECIAL_TILES["S"][0])

HOME_GRID_FIG = "phase1_home_grid.png"  # optional; use --grid_only to generate
ROUTE_EXAMPLE_BASENAME = "phase1_laundry_route_example"
SAMPLE_ROUTES_FIG = "sample_expert_routes.png"
PICKUP_COLOR = "#E67E22"


def laundry_route_example_path(
    laundry_cell: Tuple[int, int],
    results_dir: str,
) -> str:
    """Filename: phase1_laundry_route_example_{row}_{col}.png"""
    row, col = laundry_cell
    return os.path.join(results_dir, f"{ROUTE_EXAMPLE_BASENAME}_{row}_{col}.png")


def plot_home_grid_map(out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_home_grid(
        ax,
        title="12×12 Home Grid — row/col ticks match env coordinates",
        show_legend=True,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_laundry_route_example(
    start: Tuple[int, int] = DEFAULT_START,
    goal: Tuple[int, int] = DEFAULT_GOAL,
    min_laundry: int = MIN_LAUNDRY_REQUIRED,
    optimize_order: bool = True,
    out_path: str | None = None,
    *,
    results_dir: str | None = None,
    highlight_cell: Tuple[int, int] | None = None,
    target_laundry: Tuple[int, int] | None = None,
    laundry_cells: Sequence[Tuple[int, int]] | None = None,
) -> dict:
    results_dir = results_dir or os.path.join(_ROOT, "results")
    piles = list(laundry_cells) if laundry_cells is not None else list(DEFAULT_LAUNDRY)
    if target_laundry is not None:
        target = tuple(target_laundry)
        piles = [target]
        highlight_cell = highlight_cell or target
        out_path = out_path or laundry_route_example_path(target, results_dir)
    out_path = out_path or os.path.join(
        results_dir, f"{ROUTE_EXAMPLE_BASENAME}_{start[0]}_{start[1]}.png"
    )

    blocked = build_blocked_set(SPECIAL_TILES, GRID_SIZE)
    result = plan_home_route(
        start=start,
        goal=goal,
        laundry_cells=piles,
        blocked=blocked,
        min_laundry=min_laundry,
        grid_size=GRID_SIZE,
        optimize_order=optimize_order,
    )
    if result is None:
        raise RuntimeError(
            f"No route from {start} to washing machine {goal} "
            f"with {min_laundry} laundry item(s)."
        )

    path = result["positions"]
    pickup_order = result.get("pickup_order", [])

    if target_laundry is not None:
        route_title = (
            f"Expert route — start {start} → laundry bag {target_laundry} "
            f"→ washing machine {goal}"
        )
    else:
        route_title = (
            f"Expert route — start {start} → laundry bag → washing machine {goal}"
        )

    fig, ax = plt.subplots(figsize=(11, 10))
    draw_home_grid(
        ax,
        title=(
            f"{route_title}\n"
            f"{result['steps']} steps  |  pickup at {pickup_order}"
        ),
        show_legend=False,
        highlight_cell=highlight_cell,
    )
    plot_grid_path(ax, path, pickup_order=pickup_order, pickup_color=PICKUP_COLOR)

    legend_handles = [
        Line2D([0], [0], color="#2980B9", linewidth=2, label="Planned path"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PICKUP_COLOR,
               markersize=8, label="Laundry bag pickup"),
    ]
    ax.legend(
        handles=legend_patches() + legend_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result["plot_path"] = out_path
    return result


def plot_laundry_route_examples_per_bag(
    start: Tuple[int, int] = DEFAULT_START,
    goal: Tuple[int, int] = DEFAULT_GOAL,
    *,
    results_dir: str | None = None,
    min_laundry: int = MIN_LAUNDRY_REQUIRED,
) -> list[dict]:
    """One A* route figure per laundry bag on the fixed home map."""
    results_dir = results_dir or os.path.join(_ROOT, "results")
    outputs: list[dict] = []
    for bag in DEFAULT_LAUNDRY:
        result = plot_laundry_route_example(
            start=start,
            goal=goal,
            min_laundry=min_laundry,
            optimize_order=False,
            results_dir=results_dir,
            target_laundry=bag,
        )
        outputs.append(result)
        print(
            f"Laundry route example saved → {result['plot_path']}  "
            f"({result['steps']} steps, bag={bag})"
        )
    return outputs


def plot_sample_expert_routes(
    trajectories: Iterable[Sequence[Tuple[int, int]]],
    meta: Iterable[dict],
    out_path: str,
    *,
    max_panels: int = 9,
) -> str:
    traj_list = list(trajectories)[:max_panels]
    meta_list = list(meta)[:max_panels]
    n = len(traj_list)
    if n == 0:
        return out_path

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes_flat = np.atleast_1d(axes).flat

    fig.suptitle("Phase 1 — Sample Expert Home Routes", fontsize=13)

    for idx, (ax, traj) in enumerate(zip(axes_flat, traj_list)):
        # Hide fixed map start (0,0) — expert routes use random spawns, not S tile
        draw_home_grid(
            ax, show_legend=False, show_axis_labels=False, mask_cells={DEFAULT_START},
        )
        m = meta_list[idx] if idx < len(meta_list) else {}
        pickup = m.get("pickup_order") or m.get("coin_order")
        plot_grid_path(ax, traj, pickup_order=pickup, linewidth=1.2, pickup_color=PICKUP_COLOR)
        start = m.get("start", traj[0])
        ax.set_title(f"#{idx + 1}  start={start}  ({len(traj) - 1} steps)", fontsize=8)
        if idx != 0:
            ax.set_xticks([])
            ax.set_yticks([])

    for ax in axes_flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_home_figures(
    results_dir: str | None = None,
    *,
    start: Tuple[int, int] = DEFAULT_START,
    sample_trajectories: list | None = None,
    sample_meta: list | None = None,
    plot_laundry_examples: bool = False,
) -> dict:
    results_dir = results_dir or os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    paths = {}

    if plot_laundry_examples:
        examples = plot_laundry_route_examples_per_bag(
            start=start,
            results_dir=results_dir,
        )
        paths["examples"] = [ex["plot_path"] for ex in examples]

    if sample_trajectories:
        paths["samples"] = plot_sample_expert_routes(
            sample_trajectories,
            sample_meta or [],
            os.path.join(results_dir, SAMPLE_ROUTES_FIG),
        )
        print(f"Sample expert routes saved → {paths['samples']}")

    return paths


# Backward-compatible aliases
plot_escape_grid_map = plot_home_grid_map
plot_astar_example = plot_laundry_route_example
plot_sample_trajectories = plot_sample_expert_routes
generate_phase1_figures = generate_home_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise home grid + expert laundry route.")
    parser.add_argument("--start", type=int, nargs=2, default=list(DEFAULT_START), metavar=("ROW", "COL"))
    parser.add_argument("--goal", type=int, nargs=2, default=list(DEFAULT_GOAL), metavar=("ROW", "COL"))
    parser.add_argument("--min_laundry", type=int, default=MIN_LAUNDRY_REQUIRED)
    parser.add_argument("--highlight", type=int, nargs=2, default=None, metavar=("ROW", "COL"),
                        help="Highlight a grid cell on the map")
    parser.add_argument("--bag", type=int, nargs=2, default=None, metavar=("ROW", "COL"),
                        help="Plot route via one laundry bag only (default: all bags on map)")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--results_dir", default=os.path.join(_ROOT, "results"))
    parser.add_argument("--grid_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.grid_only:
        grid_path = os.path.join(args.results_dir, HOME_GRID_FIG)
        plot_home_grid_map(grid_path)
        print(f"Saved → {grid_path}")
    elif args.bag is not None:
        res = plot_laundry_route_example(
            start=tuple(args.start),
            goal=tuple(args.goal),
            min_laundry=args.min_laundry,
            optimize_order=not args.greedy,
            results_dir=args.results_dir,
            target_laundry=tuple(args.bag),
            highlight_cell=tuple(args.highlight) if args.highlight else None,
        )
        print(f"Saved → {res['plot_path']}  ({res['steps']} steps, pickup={res['pickup_order']})")
    else:
        for res in plot_laundry_route_examples_per_bag(
            start=tuple(args.start),
            goal=tuple(args.goal),
            results_dir=args.results_dir,
            min_laundry=args.min_laundry,
        ):
            print(f"Saved → {res['plot_path']}  ({res['steps']} steps, pickup={res['pickup_order']})")
