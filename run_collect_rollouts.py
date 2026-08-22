"""
run_collect_rollouts.py  —  Phase 3, Step 3 launcher

Collect successful PPO execution paths for Phase 2.5 Diffusion fine-tuning.

Usage:
    python run_collect_rollouts.py
    python run_collect_rollouts.py --min_successes 200 --device cpu
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def main():
    p = argparse.ArgumentParser(description="Step 3 — Collect successful PPO rollouts")
    p.add_argument("--min_successes", type=int, default=150)
    p.add_argument("--max_attempts", type=int, default=3000)
    p.add_argument("--ppo_ckpt", type=str, default=None)
    p.add_argument("--diffusion_ckpt", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print("   Step 3 — Collect PPO Rollouts (strict successes)")
    print("=" * 60 + "\n")

    from phase3_ppo.collect_ppo_rollouts import collect

    collect(
        min_successes=args.min_successes,
        max_attempts=args.max_attempts,
        ppo_ckpt=args.ppo_ckpt,
        diffusion_ckpt=args.diffusion_ckpt,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
