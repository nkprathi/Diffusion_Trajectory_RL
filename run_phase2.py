"""
run_phase2.py  —  Phase 2 launcher

Trains the diffusion model on expert trajectories, visualises paths, and optionally
fine-tunes on A* + PPO rollouts (Phase 2.5 via --finetune_rl).

Usage:
    python run_phase2.py
    python run_phase2.py --skip_training
    python run_phase2.py --finetune_rl
    python run_phase2.py --finetune_rl_only --skip_dataset

Prerequisite for --finetune_rl: run_collect_rollouts.py first.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def _run_finetune_rl(args) -> dict:
    from phase2_diffusion.finetune_diffusion_rl import finetune

    print("\n[Phase 2.5] Fine-tuning diffusion on A* + PPO rollouts …\n")
    return finetune(
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        lr=args.finetune_lr,
        horizon=args.finetune_horizon,
        log_every=args.finetune_log_every,
        device=args.device,
        init_ckpt=args.finetune_init_ckpt,
        rebuild_dataset=not args.finetune_skip_dataset,
    )


def main():
    p = argparse.ArgumentParser(description="Phase 2 — Train diffusion & visualise")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--schedule", type=str, default="cosine",
                   choices=["linear", "cosine", "exponential"])
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--finetune_rl", action="store_true",
                   help="After base diffusion, run Phase 2.5 RL fine-tune")
    p.add_argument("--finetune_rl_only", action="store_true",
                   help="Skip base diffusion train/viz; only run Phase 2.5 fine-tune")
    p.add_argument("--finetune_epochs", type=int, default=80)
    p.add_argument("--finetune_batch_size", type=int, default=32)
    p.add_argument("--finetune_lr", type=float, default=5e-5)
    p.add_argument("--finetune_horizon", type=int, default=16)
    p.add_argument("--finetune_log_every", type=int, default=10)
    p.add_argument("--finetune_init_ckpt", type=str, default=None)
    p.add_argument("--finetune_skip_dataset", action="store_true",
                   help="Reuse existing trajectories_rl_finetune_*.npy")
    p.add_argument("--process_plots", action="store_true",
                   help="Also save forward/reverse diffusion process figures")
    args = p.parse_args()

    print("=" * 60)
    print("   Phase 2 — Diffusion Trajectory Planner")
    print("=" * 60)

    if not args.finetune_rl_only:
        if not args.skip_training:
            print("\n[Phase 2 / Step 1] Training Temporal U-Net diffusion model …\n")
            from phase2_diffusion.train_diffusion import train
            train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                T=args.T,
                schedule=args.schedule,
                log_every=args.log_every,
                device=args.device,
                horizon=args.horizon,
            )
        else:
            print("\n[Phase 2 / Step 1] Skipping training (--skip_training).\n")

        print("\n[Phase 2 / Step 2] Generating path visualisations …\n")
        from phase2_diffusion.visualize_paths import visualize
        visualize(n_samples=args.n_samples, device=args.device)

        if args.process_plots:
            print("\n[Phase 2 / Step 3] Plotting forward & reverse diffusion processes …\n")
            from phase2_diffusion.visualisation.plot_diffusion_processes import (
                plot_diffusion_processes,
            )
            plot_diffusion_processes(device=args.device)
        else:
            print(
                "\n[Phase 2 / Step 3] Skipping forward/reverse process plots "
                "(use --process_plots to enable).\n"
            )

    if args.finetune_rl or args.finetune_rl_only:
        result = _run_finetune_rl(args)
        print(f"\n[Phase 2.5] Best val loss: {result['best_val_loss']:.5f}")
        print(f"[Phase 2.5] Checkpoint: {result['checkpoint']}")

    print("\n" + "=" * 60)
    print("   Phase 2 complete!")
    print("=" * 60)
    print("\nNext: python run_collect_rollouts.py  (if finetune not done yet)")
    print("      python run_phase3.py  →  PPO executor")


if __name__ == "__main__":
    main()
