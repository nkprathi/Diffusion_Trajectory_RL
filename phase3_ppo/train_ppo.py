"""
train_ppo.py  —  Phase 3

Train a PPO agent to follow Diffusion-generated waypoints through the escape game.

The DiffusionWaypointEnv provides:
  - A new Diffusion trajectory plan at each episode reset
  - Dense shaped reward for following waypoints
  - Pass-through of the base env rewards (goal +300, bomb -50)

PPO (Proximal Policy Optimization) uses a neural-network policy to map
20-dim observations → Discrete(4) actions, trained over 300k+ timesteps.

Outputs (to checkpoints/ and results/):
    checkpoints/ppo_best.zip        — best model by strict eval task success
    checkpoints/ppo_final.zip       — final model after all timesteps
    results/ppo_reward_curve.png    — reward and success rate over training

Usage:
    python phase3_ppo/train_ppo.py
    python phase3_ppo/train_ppo.py --timesteps 500000

References:
    Schulman et al. (2017) — PPO  — arXiv:1707.06347
    Janner et al. (2022)   — Diffuser hierarchical RL
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from phase3_ppo.env_wrapper import DiffusionWaypointEnv
from phase3_ppo.success_metrics import is_task_success

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except ImportError:
    _PLT = False


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class TrainingLogger(BaseCallback):
    """
    Records per-episode rewards and strict eval-aligned task success
    from **all** parallel sub-environments.

    Success uses ``strict_task_success`` from the env (laundry + washer + reward ≥ 250).
    Optionally stops early if rolling success drops far below its peak after min_timesteps.
    """

    def __init__(
        self,
        log_every: int = 20_000,
        total_timesteps: int = 500_000,
        early_stop: bool = True,
        min_timesteps: int = 100_000,
        success_window: int = 80,
        success_drop: float = 0.15,
        patience_logs: int = 2,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_every = log_every
        self.total_timesteps = total_timesteps
        self.early_stop = early_stop
        self.min_timesteps = min_timesteps
        self.success_window = success_window
        self.success_drop = success_drop
        self.patience_logs = patience_logs
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.successes: list[int] = []
        self.timesteps_at_done: list[int] = []
        self._ep_reward: list[float] = []
        self._ep_len: list[int] = []
        self._last_log_step = 0
        self._peak_success = 0.0
        self._bad_log_streak = 0

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._ep_reward = [0.0] * n_envs
        self._ep_len = [0] * n_envs

    def _episode_success(self, info) -> bool:
        if not isinstance(info, dict):
            return False
        return bool(info.get("strict_task_success", info.get("task_success", False)))

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i, done in enumerate(dones):
            self._ep_reward[i] += float(rewards[i])
            self._ep_len[i] += 1
            if not done:
                continue
            self.ep_rewards.append(self._ep_reward[i])
            self.ep_lengths.append(self._ep_len[i])
            self.successes.append(1 if self._episode_success(infos[i]) else 0)
            self.timesteps_at_done.append(int(self.num_timesteps))
            self._ep_reward[i] = 0.0
            self._ep_len[i] = 0

        if (
            self.num_timesteps - self._last_log_step >= self.log_every
            and len(self.ep_rewards) > 0
        ):
            recent = self.ep_rewards[-50:]
            w = min(self.success_window, len(self.successes))
            recent_s = self.successes[-w:]
            sr = float(np.mean(recent_s))
            pct = 100.0 * self.num_timesteps / max(self.total_timesteps, 1)
            print(
                f"  Timestep {self.num_timesteps:>7,d} / {self.total_timesteps:,d}  "
                f"({pct:4.1f}%)  "
                f"mean_reward={np.mean(recent):+.1f}  "
                f"strict_success={sr * 100:.1f}%  "
                f"episodes={len(self.ep_rewards)}"
            )
            self._last_log_step = self.num_timesteps

            if self.early_stop and self.num_timesteps >= self.min_timesteps:
                if sr >= self._peak_success:
                    self._peak_success = sr
                    self._bad_log_streak = 0
                elif sr < self._peak_success - self.success_drop:
                    self._bad_log_streak += 1
                else:
                    self._bad_log_streak = 0

                if self._bad_log_streak >= self.patience_logs:
                    print(
                        f"\n[train] Early stop: strict success {sr * 100:.1f}% "
                        f"fell >{self.success_drop * 100:.0f}% below peak "
                        f"{self._peak_success * 100:.1f}% "
                        f"({self.patience_logs} logs)."
                    )
                    return False

        return True


def _rolling_mean(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_indices, rolling_mean) for a 1-D array; x aligns with end of each window."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return np.arange(len(arr)), np.cumsum(arr) / np.arange(1, len(arr) + 1)
    smooth = np.convolve(arr, np.ones(window) / window, mode="valid")
    x = np.arange(len(smooth)) + window - 1
    return x, smooth


def _plot_training(
    logger: TrainingLogger,
    out_path: str,
    *,
    max_episodes: int | None = None,
    reward_smooth_window: int = 50,
    success_smooth_window: int = 50,
    show_final_annotation: bool = False,
):
    if not _PLT or len(logger.ep_rewards) < 2:
        return

    rewards = np.asarray(logger.ep_rewards, dtype=float)
    successes = np.asarray(logger.successes, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)

    if max_episodes is not None:
        rewards = rewards[:max_episodes]
        successes = successes[:max_episodes]
        episodes = episodes[:max_episodes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- Reward ---
    ax1.plot(episodes, rewards, alpha=0.12, color="steelblue", linewidth=0.6, label="Raw")
    if len(rewards) >= reward_smooth_window:
        rx, rsmooth = _rolling_mean(rewards, reward_smooth_window)
        ax1.plot(
            rx + 1,
            rsmooth,
            color="steelblue",
            linewidth=2,
            label=f"Smoothed ({reward_smooth_window} ep)",
        )
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("PPO Training — Diffusion Waypoint Follower (dynamic obstacle)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # --- Success ---
    ax2.scatter(
        episodes,
        successes * 100,
        s=6,
        alpha=0.15,
        color="darkorange",
        label="Per-episode",
        zorder=1,
    )
    sx, ssmooth = _rolling_mean(successes, success_smooth_window)
    ax2.plot(
        sx + 1,
        ssmooth * 100,
        color="darkorange",
        linewidth=2.2,
        label=f"Rolling mean ({success_smooth_window} ep)",
        zorder=2,
    )
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Completed episodes (all parallel envs)")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title(
        "Strict Task Success — matches eval metric",
        fontsize=10,
    )
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3)

    if show_final_annotation and len(ssmooth) > 0:
        ax2.axhline(ssmooth[-1] * 100, color="gray", ls="--", lw=1, alpha=0.6)
        ax2.text(
            0.98,
            0.08,
            f"Final rolling: {ssmooth[-1] * 100:.1f}%",
            transform=ax2.transAxes,
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )

    if max_episodes is not None:
        ax1.set_xlim(1, max_episodes)
        ax2.set_xlim(1, max_episodes)
    else:
        ax1.set_xlim(1, len(rewards))
        ax2.set_xlim(1, len(rewards))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Training curves saved → {out_path}")


def eval_checkpoint_success(ckpt_path: str, device: str, n_episodes: int = 30) -> float:
    """Strict eval-aligned success (laundry + washer + total_reward >= 250)."""
    from stable_baselines3 import PPO as SB3PPO

    if not os.path.exists(ckpt_path):
        return -1.0
    model = SB3PPO.load(ckpt_path, device=device)
    env = DiffusionWaypointEnv(device=device, random_start=True)
    successes = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(int(action))
            total_r += r
            done = terminated or truncated
        successes.append(is_task_success(env.base_env, total_reward=total_r))
    env.close()
    return float(np.mean(successes))


def collect_checkpoint_candidates(checkpoint_dir: str) -> list[str]:
    """All PPO .zip checkpoints in deterministic evaluation order."""
    names = [
        "ppo_best.zip",
        "best_model.zip",
        "ppo_final.zip",
    ]
    candidates = [
        os.path.join(checkpoint_dir, n)
        for n in names
        if os.path.isfile(os.path.join(checkpoint_dir, n))
    ]
    ckpt_glob = sorted(glob.glob(os.path.join(checkpoint_dir, "ppo_ckpt_*_steps.zip")))
    seen = set(candidates)
    for path in ckpt_glob:
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    return candidates


def select_best_checkpoint(
    candidates: list[str],
    device: str,
    out_path: str,
    n_episodes: int = 30,
) -> tuple[str, float]:
    """Copy the checkpoint with highest strict eval success to out_path."""
    best_path, best_sr = "", -1.0
    for path in candidates:
        if not os.path.exists(path):
            continue
        sr = eval_checkpoint_success(path, device, n_episodes=n_episodes)
        label = os.path.basename(path)
        print(f"  [select] {label:30s}  strict_success={sr * 100:5.1f}%")
        if sr > best_sr:
            best_sr, best_path = sr, path
    if best_path:
        if os.path.abspath(best_path) != os.path.abspath(out_path):
            shutil.copy2(best_path, out_path)
            print(f"[train] Selected {os.path.basename(best_path)} → {out_path}  ({best_sr * 100:.1f}%)")
        else:
            print(f"[train] Best checkpoint already at {out_path}  ({best_sr * 100:.1f}%)")
    return best_path, best_sr


# Backward-compatible aliases
_eval_checkpoint_success = eval_checkpoint_success
_select_best_checkpoint = select_best_checkpoint


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(
    timesteps:      int   = 500_000,
    lr:             float = 3e-4,
    n_steps:        int   = 2048,
    batch_size:     int   = 64,
    n_epochs:       int   = 10,
    gamma:          float = 0.99,
    ent_coef:       float = 0.01,
    log_every:      int   = 20_000,
    checkpoint_dir: str   = None,
    results_dir:    str   = None,
    device:         str   = "cpu",
    resume_path:    str | None = None,
    save_freq:      int   = 100_000,
    early_stop:     bool  = True,
    select_n_episodes: int = 50,
):
    checkpoint_dir = checkpoint_dir or os.path.join(_ROOT, "checkpoints")
    results_dir    = results_dir    or os.path.join(_ROOT, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    n_envs = 4
    steps_per_rollout = n_steps * n_envs

    print(f"[train] Device: {device}")
    print(f"[train] Target timesteps: {timesteps:,}  ({n_envs} parallel envs)")
    print(f"[train] Rollout size: {steps_per_rollout:,} env-steps  "
          f"(~{timesteps // steps_per_rollout} PPO updates)")

    # --- Environment -------------------------------------------------------
    def _make_env():
        env = DiffusionWaypointEnv(device=device, random_start=True)
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env] * n_envs)

    # --- PPO model ---------------------------------------------------------
    if resume_path and os.path.exists(resume_path):
        print(f"[train] Resuming from {resume_path}")
        model = PPO.load(resume_path, env=vec_env, device=device)
    else:
        if resume_path:
            print(f"[train] Resume path not found ({resume_path}) — training from scratch")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            ent_coef=ent_coef,
            verbose=0,
            device=device,
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"[model] PPO policy parameters: {n_params:,}")

    # --- Callbacks ---------------------------------------------------------
    logger = TrainingLogger(
        log_every=log_every,
        total_timesteps=timesteps,
        early_stop=early_stop,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="ppo_ckpt",
        verbose=0,
    )

    # --- Train -------------------------------------------------------------
    print(f"\n[train] Starting PPO training …")
    print(f"[train] Strict eval metric: laundry + washer + reward ≥ 250\n")
    model.learn(
        total_timesteps=timesteps,
        callback=[logger, ckpt_cb],
        progress_bar=False,
        reset_num_timesteps=not bool(resume_path and os.path.exists(resume_path)),
    )

    completed = model.num_timesteps
    print(f"\n[train] Completed {completed:,} / {timesteps:,} timesteps")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "ppo_final")
    model.save(final_path)
    print(f"[train] Final weights → {final_path}.zip")

    # Select best by strict eval success (not EvalCallback mean reward)
    best_dst = os.path.join(checkpoint_dir, "ppo_best.zip")
    ckpt_candidates = collect_checkpoint_candidates(checkpoint_dir)
    ckpt_candidates.append(f"{final_path}.zip")
    ckpt_candidates = list(dict.fromkeys(ckpt_candidates))  # dedupe, preserve order
    print(f"\n[train] Selecting best checkpoint by strict eval success "
          f"({select_n_episodes} episodes) …")
    select_best_checkpoint(
        ckpt_candidates, device, best_dst, n_episodes=select_n_episodes,
    )

    # --- Plots -------------------------------------------------------------
    metrics_path = os.path.join(results_dir, "ppo_training_metrics.npz")
    np.savez(
        metrics_path,
        ep_rewards=np.array(logger.ep_rewards, dtype=np.float32),
        successes=np.array(logger.successes, dtype=np.int8),
        ep_lengths=np.array(logger.ep_lengths, dtype=np.int32),
        timesteps_at_done=np.array(logger.timesteps_at_done, dtype=np.int32),
    )
    _plot_training(logger, os.path.join(results_dir, "ppo_reward_curve.png"))

    # Summary stats
    if len(logger.successes) > 0:
        final_sr = np.mean(logger.successes[-100:]) * 100
        print(f"[train] Final success rate (last 100 ep): {final_sr:.1f}%")

    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="Train PPO with Diffusion waypoints")
    p.add_argument("--timesteps",  type=int,   default=500_000)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--n_steps",    type=int,   default=2048)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--log_every",  type=int,   default=20_000)
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--resume",     type=str,   default=None,
                   help="Resume from a .zip checkpoint (e.g. checkpoints/ppo_final.zip)")
    p.add_argument("--save_freq",  type=int,   default=100_000,
                   help="Save intermediate ppo_ckpt_* every N timesteps")
    p.add_argument("--no_early_stop", action="store_true",
                   help="Disable early stop on success plateau")
    p.add_argument("--select_n_episodes", type=int, default=50,
                   help="Episodes per checkpoint when selecting ppo_best")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(
        timesteps=args.timesteps,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        log_every=args.log_every,
        device=args.device,
        resume_path=args.resume,
        save_freq=args.save_freq,
        early_stop=not args.no_early_stop,
        select_n_episodes=args.select_n_episodes,
    )
