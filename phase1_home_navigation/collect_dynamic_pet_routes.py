"""
collect_dynamic_pet_routes.py — Phase 2 diffusion data

Generate A* routes with random dynamic-PET cells blocked during planning.
Teaches detours around local obstacles (PET-aware expert data).

Output:
    data/raw_trajectories_dynamic_pet.npy
    data/raw_trajectories_dynamic_pet_meta.npy
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from env.map_config import MIN_LAUNDRY_REQUIRED, SPECIAL_TILES
from phase1_home_navigation.home_route_planner import (
    build_blocked_set,
    list_empty_cells,
    plan_home_route,
)
from phase1_home_navigation.collect_expert_routes import (
    DEFAULT_GOAL,
    DEFAULT_LAUNDRY,
    GRID_SIZE,
    _sample_start,
)


def _walkable_cells(blocked: set, grid_size: int = GRID_SIZE) -> list[tuple[int, int]]:
    cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) in blocked:
                continue
            cells.append((r, c))
    return cells


def collect_dynamic_pet_routes(
    n_trajectories: int = 200,
    *,
    seed: int = 42,
    out_dir: str | None = None,
) -> tuple[list, list]:
    rng = random.Random(seed)
    blocked_static = build_blocked_set(SPECIAL_TILES, GRID_SIZE)
    empty_cells = list_empty_cells(SPECIAL_TILES, GRID_SIZE, blocked_static)
    walkable = _walkable_cells(blocked_static)

    trajectories, meta = [], []
    attempts = 0
    max_attempts = max(n_trajectories * 80, 3000)

    while len(trajectories) < n_trajectories and attempts < max_attempts:
        attempts += 1
        start = _sample_start(empty_cells, blocked_static, rng, random_start=True)
        if start is None:
            break

        forbidden = blocked_static | {start, DEFAULT_GOAL}
        for lc in DEFAULT_LAUNDRY:
            forbidden.add(tuple(lc))
        pet_candidates = [c for c in walkable if c not in forbidden]
        if not pet_candidates:
            continue
        n_pet = rng.randint(1, min(2, len(pet_candidates)))
        pet_blocked = set(rng.sample(pet_candidates, n_pet))
        blocked_plan = blocked_static | pet_blocked

        result = plan_home_route(
            start=start,
            goal=DEFAULT_GOAL,
            laundry_cells=DEFAULT_LAUNDRY,
            blocked=blocked_plan,
            min_laundry=MIN_LAUNDRY_REQUIRED,
            grid_size=GRID_SIZE,
            optimize_order=True,
        )
        if result is None:
            continue

        trajectories.append(result["positions"])
        meta.append({
            "start": start,
            "goal": DEFAULT_GOAL,
            "pet_blocked": list(pet_blocked),
            "pickup_order": result.get("pickup_order", []),
            "laundry_collected": result["laundry_collected"],
            "steps": result["steps"],
            "planner": "home_route_astar_pet_aware",
        })

        if len(trajectories) % 50 == 0:
            print(f"  dynamic PET routes: {len(trajectories)}/{n_trajectories}")

    return trajectories, meta


def main():
    p = argparse.ArgumentParser(description="Collect PET-aware A* expert routes")
    p.add_argument("--n_trajectories", type=int, default=200)
    p.add_argument("--out_dir", type=str, default=os.path.join(_ROOT, "data"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[collect] Dynamic-PET expert routes (n={args.n_trajectories}) …")
    trajs, meta = collect_dynamic_pet_routes(
        args.n_trajectories, seed=args.seed, out_dir=args.out_dir,
    )
    if not trajs:
        sys.exit("[collect] No routes generated.")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "raw_trajectories_dynamic_pet.npy")
    meta_path = os.path.join(args.out_dir, "raw_trajectories_dynamic_pet_meta.npy")
    np.save(path, np.array(trajs, dtype=object), allow_pickle=True)
    np.save(meta_path, np.array(meta, dtype=object), allow_pickle=True)
    print(f"[collect] Saved {len(trajs)} routes → {path}")
    print(f"[collect] Meta → {meta_path}")


if __name__ == "__main__":
    main()
