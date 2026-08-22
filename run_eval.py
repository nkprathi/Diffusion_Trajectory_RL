"""
run_eval.py — Diffusion planning evaluation (+ optional moving-obstacle demo)

    Diffusion open-loop | Diffusion + PPO | Diffusion-RL open-loop | Diffusion-RL + PPO

Usage:
    python run_eval.py
    python run_eval.py --include_astar
    python run_eval.py --render_demo
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def main():
    p = argparse.ArgumentParser(description="Run diffusion planning evaluation")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--periodic_replan",
        type=int,
        default=-1,
        help="Optional fixed replan interval (-1 = event-driven only)",
    )
    p.add_argument("--ppo_ckpt", type=str, default=None)
    p.add_argument("--diffusion_ckpt", type=str, default=None)
    p.add_argument("--diffusion_rl_ckpt", type=str, default=None)
    p.add_argument("--baseline_only", action="store_true",
                   help="Skip Diffusion-RL methods (2 diffusion methods only)")
    p.add_argument("--include_astar", action="store_true",
                   help="Include A* open-loop and A* + replan")
    p.add_argument("--render_demo", action="store_true",
                   help="Render moving-obstacle demo video after eval")
    args = p.parse_args()

    from phase3_ppo.evaluate import evaluate

    ckpt_dir = os.path.join(_ROOT, "checkpoints")
    diffusion_rl = args.diffusion_rl_ckpt or os.path.join(ckpt_dir, "diffusion_rl_finetuned.pt")

    print("=" * 60)
    print("   Diffusion Planning Evaluation")
    print("=" * 60)

    periodic = None if args.periodic_replan < 0 else args.periodic_replan
    evaluate(
        n_episodes=args.n_episodes,
        device=args.device,
        periodic_replan=periodic,
        ppo_ckpt=args.ppo_ckpt,
        diffusion_ckpt=args.diffusion_ckpt,
        diffusion_rl_ckpt=diffusion_rl,
        include_rl_finetuned=not args.baseline_only,
        diffusion_only=not args.include_astar,
    )

    if args.render_demo:
        print("\n" + "=" * 60)
        print("   Moving Obstacle Demo (Diffusion-RL + PPO)")
        print("=" * 60)
        from phase3_ppo.render_moving_obstacle import render_moving_obstacle
        render_moving_obstacle(
            device=args.device,
            ppo_ckpt=args.ppo_ckpt,
            diffusion_ckpt=diffusion_rl if os.path.isfile(diffusion_rl) else args.diffusion_ckpt,
        )


if __name__ == "__main__":
    main()
