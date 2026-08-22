"""
run_phase1.py — One-command launcher for Phase 1 (home navigation)

Pipeline:
  Step 1: Collect expert routes (laundry → washing machine) → data/raw_trajectories.npy
  Step 2: Prepare Diffusion dataset                         → data/trajectories_train.npy
  Step 3: Route figures                               → results/phase1_laundry_route_example_{row}_{col}.png
                                                       → results/sample_expert_routes.png

Usage (from project root):
    python run_phase1.py
    python run_phase1.py --n_trajectories 800
    python run_phase1.py --viz_only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run all Phase 1 steps.")
    parser.add_argument("--n_trajectories", type=int, default=600)
    parser.add_argument(
        "--min_laundry", type=int, default=None,
        help="Laundry pickups required (default: MIN_LAUNDRY_REQUIRED from map)",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--fixed_start", action="store_true",
                        help="Use fixed start (0,0) instead of random spawns")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy laundry selection (faster collection)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz_only", action="store_true",
                        help="Only regenerate Phase 1 home grid figures")
    args = parser.parse_args()

    from env.map_config import MIN_LAUNDRY_REQUIRED
    min_laundry = args.min_laundry if args.min_laundry is not None else MIN_LAUNDRY_REQUIRED

    if args.viz_only:
        print("=" * 55)
        print("Phase 1 — Home grid figures only")
        print("=" * 55)
        from phase1_home_navigation.visualize_home_grid import generate_home_figures
        generate_home_figures()
    else:
        print("=" * 55)
        print("STEP 1 — Collecting expert home routes")
        print("  (start → laundry → washing machine)")
        print("=" * 55)
        from phase1_home_navigation.collect_expert_routes import collect
        collect(
            n_trajectories=args.n_trajectories,
            min_laundry=min_laundry,
            random_start=not args.fixed_start,
            optimize_order=not args.greedy,
            out_dir="data",
            seed=args.seed,
        )

        print("\n" + "=" * 55)
        print("STEP 2 — Preparing Diffusion trajectory dataset")
        print("=" * 55)
        from phase1_home_navigation.prepare_trajectory_dataset import prepare
        prepare(
            raw_path="data/raw_trajectories.npy",
            out_dir="data",
            horizon=args.horizon,
        )

    print("\n" + "=" * 55)
    print("PHASE 1 COMPLETE")
    print("=" * 55)
    print("Files ready:")
    print("  data/raw_trajectories.npy")
    print("  data/trajectories_train.npy   ← feed this to Phase 2 diffusion")
    print("  data/trajectories_val.npy")
    print("  results/phase1_laundry_route_example_1_11.png")
    print("  results/phase1_laundry_route_example_7_3.png")
    print("  results/sample_expert_routes.png")
    print("  results/data_distribution.png")
    print("\nNext:  python run_phase2.py  to train the diffusion model.")


if __name__ == "__main__":
    main()
