"""
collect_ppo_rollouts.py  —  Phase 3, Step 3

Roll out the trained PPO agent in DiffusionWaypointEnv and save **successful**
executed paths for Phase 2.5 (RL-informed diffusion fine-tuning).

Only episodes with strict task success (laundry + washer tile) are kept.

Outputs (data/):
    rl_trajectories.npy       — object array of (T, 2) grid paths, same layout as
                                raw_trajectories.npy from Phase 1
    rl_trajectories_meta.npy  — per-path metadata (start, steps, reward, …)

Usage:
    python phase3_ppo/collect_ppo_rollouts.py
    python run_collect_rollouts.py --min_successes 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env.map_config import MIN_LAUNDRY_REQUIRED, SPECIAL_TILES
from phase2_diffusion.checkpoint_names import DIFFUSION_BEST, resolve_checkpoint
from phase3_ppo.env_wrapper import DiffusionWaypointEnv, MAX_STEPS
from phase3_ppo.success_metrics import is_task_success

DEFAULT_GOAL = tuple(SPECIAL_TILES["G"][0])
RL_TRAJ_FILE = "rl_trajectories.npy"
RL_META_FILE = "rl_trajectories_meta.npy"


def _path_to_tuples(path: np.ndarray) -> list[tuple[int, int]]:
    """Convert executed path array to integer (row, col) tuples."""
    out: list[tuple[int, int]] = []
    for row, col in path:
        cell = (int(row), int(col))
        if not out or out[-1] != cell:
            out.append(cell)
    return out


def collect(
    *,
    min_successes: int = 150,
    max_attempts: int = 3000,
    ppo_ckpt: str | None = None,
    diffusion_ckpt: str | None = None,
    out_dir: str | None = None,
    device: str = "cpu",
    seed: int = 42,
    deterministic: bool = True,
) -> dict:
    from stable_baselines3 import PPO as SB3PPO

    out_dir = out_dir or os.path.join(_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)

    ppo_ckpt = ppo_ckpt or os.path.join(_ROOT, "checkpoints", "ppo_best.zip")
    diffusion_ckpt = diffusion_ckpt or resolve_checkpoint(
        os.path.join(_ROOT, "checkpoints"), DIFFUSION_BEST
    )

    if not os.path.isfile(ppo_ckpt):
        sys.exit(f"PPO checkpoint not found: {ppo_ckpt}\nRun Phase 3 training first.")
    if not os.path.isfile(diffusion_ckpt):
        sys.exit(f"Diffusion checkpoint not found: {diffusion_ckpt}\nRun Phase 2 first.")

    print(f"[rollout] Loading PPO from {ppo_ckpt}")
    model = SB3PPO.load(ppo_ckpt, device=device)
    env = DiffusionWaypointEnv(ckpt_path=diffusion_ckpt, device=device, random_start=True)

    np.random.seed(seed)
    trajectories: list[list[tuple[int, int]]] = []
    meta: list[dict] = []
    attempts = 0
    t0 = time.time()

    print(
        f"[rollout] Collecting ≥{min_successes} successful paths "
        f"(max {max_attempts} attempts, dynamic PET) …"
    )

    while len(trajectories) < min_successes and attempts < max_attempts:
        attempts += 1
        obs, info = env.reset()
        start = (int(env.base_env.state[0]), int(env.base_env.state[1]))
        total_r = 0.0
        steps = 0
        path_cells = [[env.base_env.state[0], env.base_env.state[1]]]
        done = False

        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_r += float(r)
            steps += 1
            path_cells.append([env.base_env.state[0], env.base_env.state[1]])

        success = is_task_success(env.base_env, total_reward=total_r)
        if not success:
            continue

        traj = _path_to_tuples(np.array(path_cells, dtype=np.float32))
        if len(traj) < 2:
            continue

        trajectories.append(traj)
        meta.append({
            "start": start,
            "goal": DEFAULT_GOAL,
            "steps": len(traj) - 1,
            "episode_reward": total_r,
            "laundry_collected": sum(env.base_env.coins),
            "waypoints_done": int(info.get("waypoints_done", 0)),
            "replanned_after_laundry": bool(info.get("replanned_after_laundry", False)),
            "planner": "diffusion_ppo_rollout",
            "ppo_ckpt": os.path.basename(ppo_ckpt),
            "diffusion_ckpt": os.path.basename(diffusion_ckpt),
        })

        if len(trajectories) % 25 == 0:
            rate = len(trajectories) / attempts * 100
            print(
                f"  saved {len(trajectories):4d}/{min_successes}  "
                f"(attempt {attempts}, hit_rate={rate:.1f}%, last_len={len(traj)-1})"
            )

    env.close()

    if not trajectories:
        sys.exit(
            f"No successful rollouts in {attempts} attempts. "
            "Check ppo_best.zip or lower --min_successes."
        )

    traj_path = os.path.join(out_dir, RL_TRAJ_FILE)
    meta_path = os.path.join(out_dir, RL_META_FILE)
    np.save(traj_path, np.array(trajectories, dtype=object), allow_pickle=True)
    np.save(meta_path, np.array(meta, dtype=object), allow_pickle=True)

    elapsed = time.time() - t0
    avg_steps = float(np.mean([m["steps"] for m in meta]))
    hit_rate = len(trajectories) / max(attempts, 1) * 100

    summary = {
        "n_success": len(trajectories),
        "n_attempts": attempts,
        "hit_rate_pct": hit_rate,
        "avg_steps": avg_steps,
        "traj_path": traj_path,
        "meta_path": meta_path,
        "elapsed_s": elapsed,
    }

    print(f"\n[rollout] Collected {len(trajectories)} successful paths in {attempts} attempts.")
    print(f"  Hit rate     : {hit_rate:.1f}%")
    print(f"  Avg path len : {avg_steps:.1f} steps")
    print(f"  Elapsed      : {elapsed:.1f}s")
    print(f"  Saved → {traj_path}")
    print(f"  Meta  → {meta_path}")
    print("\nNext: python run_phase2.py --finetune_rl  (merge A* + RL data, fine-tune diffusion)")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect successful PPO rollout trajectories.")
    parser.add_argument("--min_successes", type=int, default=150)
    parser.add_argument("--max_attempts", type=int, default=3000)
    parser.add_argument("--ppo_ckpt", type=str, default=None)
    parser.add_argument("--diffusion_ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic PPO actions (default: deterministic)")
    args = parser.parse_args()

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
