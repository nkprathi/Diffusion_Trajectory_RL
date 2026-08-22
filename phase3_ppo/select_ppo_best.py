"""
select_ppo_best.py — Phase 1

Evaluate all PPO checkpoints with the same strict metric as run_eval.py
(laundry + washer tile + total_reward >= 250) and copy the best to ppo_best.zip.

Usage:
    python phase3_ppo/select_ppo_best.py
    python phase3_ppo/select_ppo_best.py --n_episodes 50 --device cpu
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from phase3_ppo.train_ppo import collect_checkpoint_candidates, select_best_checkpoint


def main():
    p = argparse.ArgumentParser(description="Select ppo_best.zip by strict eval success")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--out", type=str, default=None,
                   help="Output path (default: checkpoints/ppo_best.zip)")
    args = p.parse_args()

    ckpt_dir = args.checkpoint_dir or os.path.join(_ROOT, "checkpoints")
    out_path = args.out or os.path.join(ckpt_dir, "ppo_best.zip")
    candidates = collect_checkpoint_candidates(ckpt_dir)

    print(f"[select] Evaluating {len(candidates)} checkpoints "
          f"({args.n_episodes} episodes each, strict metric) …\n")
    best_path, best_sr = select_best_checkpoint(
        candidates,
        device=args.device,
        out_path=out_path,
        n_episodes=args.n_episodes,
    )
    if not best_path:
        print("[select] No checkpoints found.")
        sys.exit(1)
    print(f"\n[select] Best: {os.path.basename(best_path)}  ({best_sr * 100:.1f}%)")


if __name__ == "__main__":
    main()
