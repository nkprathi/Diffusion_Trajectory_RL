"""
run_phase3.py  —  Phase 3 launcher (PPO executor)

Trains PPO on Diffusion waypoints, optionally re-selects best checkpoint,
and runs evaluation.

Usage:
    python run_phase3.py
    python run_phase3.py --skip_training
    python run_phase3.py --reselect_before_train --timesteps 200000 --diffusion_only_eval
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def main():
    p = argparse.ArgumentParser(description="Phase 3 — Train PPO + Evaluate")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=20_000)
    p.add_argument("--n_eval_ep", type=int, default=100)
    p.add_argument("--select_n_episodes", type=int, default=50,
                   help="Episodes per checkpoint when selecting ppo_best")
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--reselect_before_train", action="store_true",
                   help="Pick best existing ppo_best before training")
    p.add_argument("--early_stop", action="store_true",
                   help="Stop PPO when strict success plateaus (recommended for shorter runs)")
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--diffusion_only_eval", action="store_true",
                   help="Eval 4 diffusion methods only (no A* bars)")
    args = p.parse_args()

    early_stop = args.early_stop and not args.no_early_stop
    ckpt_dir = os.path.join(_ROOT, "checkpoints")
    best_dst = os.path.join(ckpt_dir, "ppo_best.zip")

    print("=" * 60)
    print("   Phase 3 — PPO Executor with Diffusion Waypoint Planner")
    print("=" * 60)

    from phase3_ppo.train_ppo import (
        collect_checkpoint_candidates,
        select_best_checkpoint,
        train,
    )

    if args.reselect_before_train or args.skip_training:
        print("\n[Phase 3] Selecting best existing checkpoint (strict metric) …\n")
        select_best_checkpoint(
            collect_checkpoint_candidates(ckpt_dir),
            device=args.device,
            out_path=best_dst,
            n_episodes=args.select_n_episodes,
        )

    if not args.skip_training:
        print("\n[Phase 3 / Step 1] Training PPO agent …\n")
        train(
            timesteps=args.timesteps,
            lr=args.lr,
            log_every=args.log_every,
            device=args.device,
            resume_path=args.resume,
            early_stop=early_stop,
            select_n_episodes=args.select_n_episodes,
        )
    else:
        print("\n[Phase 3 / Step 1] Skipping training (--skip_training).\n")

    if not args.skip_eval:
        print("\n[Phase 3 / Step 2] Running evaluation …\n")
        from phase3_ppo.evaluate import evaluate
        evaluate(
            n_episodes=args.n_eval_ep,
            device=args.device,
            diffusion_only=args.diffusion_only_eval,
        )
    else:
        print("\n[Phase 3 / Step 2] Skipping eval (--skip_eval).\n")

    print("\n" + "=" * 60)
    print("   Phase 3 complete!")
    print("=" * 60)
    print("  checkpoints/ppo_best.zip")
    print("  results/ppo_reward_curve.png")
    print("  results/eval_success_rate.png")
    print("  results/eval_method_comparison.png")


if __name__ == "__main__":
    main()
