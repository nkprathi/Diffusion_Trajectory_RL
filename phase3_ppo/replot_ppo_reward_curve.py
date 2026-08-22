"""
Replot ppo_reward_curve.png from saved training metrics.

Usage:
    python phase3_ppo/replot_ppo_reward_curve.py
    python phase3_ppo/replot_ppo_reward_curve.py --max_episodes 500
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from phase3_ppo.train_ppo import TrainingLogger, _plot_training


def main():
    p = argparse.ArgumentParser(description="Replot PPO training curve from metrics npz")
    p.add_argument(
        "--metrics",
        type=str,
        default=os.path.join(_ROOT, "results", "ppo_training_metrics.npz"),
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join(_ROOT, "results", "ppo_reward_curve.png"),
    )
    p.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Plot only the first N completed episodes (default: all)",
    )
    p.add_argument("--reward_smooth_window", type=int, default=50)
    p.add_argument("--success_smooth_window", type=int, default=50)
    p.add_argument("--show_final_annotation", action="store_true")
    args = p.parse_args()

    if not os.path.isfile(args.metrics):
        sys.exit(
            f"Missing {args.metrics}\n"
            "Run PPO training once to save metrics, or use an existing npz."
        )

    data = np.load(args.metrics)
    logger = TrainingLogger()
    logger.ep_rewards = data["ep_rewards"].tolist()
    logger.successes = data["successes"].tolist()
    logger.ep_lengths = data["ep_lengths"].tolist()
    if "timesteps_at_done" in data:
        logger.timesteps_at_done = data["timesteps_at_done"].tolist()

    n = len(logger.ep_rewards)
    sr_all = np.mean(logger.successes) * 100 if n else 0.0
    sr_last = np.mean(logger.successes[-100:]) * 100 if n >= 100 else sr_all
    print(f"[replot] {n} episodes  overall_success={sr_all:.1f}%  last100={sr_last:.1f}%")

    _plot_training(
        logger,
        args.out,
        max_episodes=args.max_episodes,
        reward_smooth_window=args.reward_smooth_window,
        success_smooth_window=args.success_smooth_window,
        show_final_annotation=args.show_final_annotation,
    )


if __name__ == "__main__":
    main()
