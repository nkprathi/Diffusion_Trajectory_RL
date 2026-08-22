"""
collect_expert_routes.py — Phase 1: collect A* expert routes on the home grid

Generate expert trajectories:
  start → any laundry bag → washing machine

Usage:
    python phase1_home_navigation/collect_expert_routes.py
    python phase1_home_navigation/collect_expert_routes.py --n_trajectories 600

Output:
    data/raw_trajectories.npy
    data/raw_trajectories_meta.npy
    results/phase1_laundry_route_example_{row}_{col}.png  (one per laundry bag)
    results/sample_expert_routes.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.map_config import MIN_LAUNDRY_REQUIRED, SPECIAL_TILES
from phase1_home_navigation.home_route_planner import (
    build_blocked_set,
    list_empty_cells,
    plan_home_route,
)
from phase1_home_navigation.visualize_home_grid import (
    DEFAULT_GOAL,
    DEFAULT_START,
    generate_home_figures,
)

GRID_SIZE = 12
DEFAULT_LAUNDRY = [tuple(c) for c in SPECIAL_TILES["LB"]]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sample_start(empty_cells, blocked, rng, *, random_start):
    if random_start and empty_cells:
        return rng.choice(empty_cells)
    if DEFAULT_START not in blocked:
        return DEFAULT_START
    return rng.choice(empty_cells) if empty_cells else None


def collect(
    n_trajectories: int = 600,
    min_laundry: int = MIN_LAUNDRY_REQUIRED,
    random_start: bool = True,
    optimize_order: bool = True,
    out_dir: str = "../data",
    results_dir: str | None = None,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    blocked = build_blocked_set(SPECIAL_TILES, GRID_SIZE)
    empty_cells = list_empty_cells(SPECIAL_TILES, GRID_SIZE, blocked)
    goal = DEFAULT_GOAL
    results_dir = results_dir or os.path.join(_ROOT, "results")

    print(f"Home route collector  grid={GRID_SIZE}×{GRID_SIZE}")
    print(f"  Washing machine: {goal}  laundry bags: {len(DEFAULT_LAUNDRY)}  min_pickups: {min_laundry}")
    print(f"  Random start: {random_start}  empty spawns: {len(empty_cells)}")

    trajectories, meta = [], []
    attempts, max_attempts = 0, max(n_trajectories * 50, 5000)

    while len(trajectories) < n_trajectories and attempts < max_attempts:
        attempts += 1
        start = _sample_start(empty_cells, blocked, rng, random_start=random_start)
        if start is None:
            sys.exit("No valid start cell found on grid.")

        result = plan_home_route(
            start=start,
            goal=goal,
            laundry_cells=DEFAULT_LAUNDRY,
            blocked=blocked,
            min_laundry=min_laundry,
            grid_size=GRID_SIZE,
            optimize_order=optimize_order,
        )
        if result is None:
            continue

        trajectories.append(result["positions"])
        meta.append({
            "start": start,
            "goal": goal,
            "laundry_collected": result["laundry_collected"],
            "pickup_order": result.get("pickup_order", []),
            "steps": result["steps"],
            "planner": "home_route_astar",
        })

        if len(trajectories) % 50 == 0:
            print(
                f"  collected {len(trajectories):4d}/{n_trajectories} "
                f"(attempt {attempts}, last steps={result['steps']})"
            )

    if not trajectories:
        sys.exit("Planner failed to produce any routes. Check map / blocked tiles.")

    print(f"\nCollected {len(trajectories)} routes in {attempts} attempts.")
    print(f"  Avg steps : {np.mean([m['steps'] for m in meta]):.1f}")
    print(f"  Avg pickups: {np.mean([m['laundry_collected'] for m in meta]):.1f}")

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_trajectories.npy")
    np.save(raw_path, np.array(trajectories, dtype=object), allow_pickle=True)
    print(f"Raw trajectories saved → {raw_path}")

    meta_path = os.path.join(out_dir, "raw_trajectories_meta.npy")
    np.save(meta_path, np.array(meta, dtype=object), allow_pickle=True)
    print(f"Metadata saved         → {meta_path}")

    print("\nGenerating Phase 1 home grid figures …")
    generate_home_figures(
        results_dir=results_dir,
        start=DEFAULT_START,
        sample_trajectories=trajectories[:9],
        sample_meta=meta[:9],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect home-navigation expert routes.")
    parser.add_argument("--n_trajectories", type=int, default=600)
    parser.add_argument(
        "--min_laundry", type=int, default=MIN_LAUNDRY_REQUIRED,
        help="Laundry items required before washing machine (default: 1)",
    )
    parser.add_argument("--fixed_start", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--out_dir", default=os.path.join(_ROOT, "data"))
    parser.add_argument("--results_dir", default=os.path.join(_ROOT, "results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collect(
        n_trajectories=args.n_trajectories,
        min_laundry=args.min_laundry,
        random_start=not args.fixed_start,
        optimize_order=not args.greedy,
        out_dir=args.out_dir,
        results_dir=args.results_dir,
        seed=args.seed,
    )
