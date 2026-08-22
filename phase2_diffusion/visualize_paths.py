"""
visualize_paths.py  —  Phase 2

Load the best Diffusion checkpoint, generate trajectories, and create plots:
    1. results/diffusion_samples_grid.png  — grid of 16 generated paths
    2. results/diffusion_vs_expert.png     — comparison of expert vs generated
    3. results/denoising_chain.png    — one trajectory across denoising steps

Usage:
    python phase2_diffusion/visualize_paths.py
    python phase2_diffusion/visualize_paths.py --n_samples 16
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from phase2_diffusion.noise_schedule import NoiseSchedule
from phase2_diffusion.unet import TemporalUNet
from phase2_diffusion.diffusion import GaussianDiffusion
from phase2_diffusion.trajectory_utils import (
    make_context,
    make_task_context,
    goal_normalized,
    normalize_grid,
    nearest_laundry_cell,
    TASK_CONTEXT_DIM,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


GRID_SIZE = 12


def _denorm(coords: np.ndarray) -> np.ndarray:
    """[-1,1] → [0, GRID_SIZE-1]"""
    return (coords + 1.0) / 2.0 * (GRID_SIZE - 1)


def _build_sample_context(
    starts: torch.Tensor,
    goal_t: torch.Tensor,
    context_dim: int,
    device: str,
) -> torch.Tensor:
    if context_dim >= TASK_CONTEXT_DIM:
        laundry_list = []
        starts_np = starts.cpu().numpy()
        for i in range(starts_np.shape[0]):
            sr = int(np.clip(round(_denorm(starts_np[i : i + 1])[0][0]), 0, GRID_SIZE - 1))
            sc = int(np.clip(round(_denorm(starts_np[i : i + 1])[0][1]), 0, GRID_SIZE - 1))
            laundry_list.append(normalize_grid(nearest_laundry_cell([sr, sc])))
        laundry_t = torch.tensor(np.stack(laundry_list), device=device, dtype=torch.float32)
        return make_task_context(starts, goal_t, laundry_t)
    return make_context(starts, goal_t)


def _load_model(ckpt_path: str, device: str) -> tuple[TemporalUNet, GaussianDiffusion, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]

    noise_sched = NoiseSchedule(
        T=cfg["T"],
        schedule=cfg["schedule"],
        device=device,
    )
    context_dim = cfg.get("context_dim", 0)
    unet = TemporalUNet(
        state_dim=cfg["state_dim"],
        horizon=cfg["horizon"],
        channels=tuple(cfg["channels"]),
        time_embed_dim=cfg["time_embed_dim"],
        context_dim=context_dim,
    )
    unet.load_state_dict(ckpt["model_state"])
    unet.eval()

    diffusion = GaussianDiffusion(unet, noise_sched, device=device)
    print(f"[vis] Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")
    return unet, diffusion, cfg


# ---------------------------------------------------------------------------
# Plot 1 — Grid of generated trajectories
# ---------------------------------------------------------------------------

def plot_samples_grid(
    trajs_norm: torch.Tensor,
    out_path: str,
    n: int = 16,
    title: str = "Diffusion Generated Trajectories",
):
    trajs = _denorm(trajs_norm.cpu().numpy())[:n]
    cols = 4
    rows = (len(trajs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    colors = cm.viridis(np.linspace(0, 1, trajs.shape[1]))

    for i, traj in enumerate(trajs):
        ax = axes[i]
        # Draw grid lines
        for k in range(GRID_SIZE + 1):
            ax.axhline(k - 0.5, color="lightgray", linewidth=0.4)
            ax.axvline(k - 0.5, color="lightgray", linewidth=0.4)

        # Colour trajectory by waypoint index (progress)
        for w in range(len(traj) - 1):
            ax.plot(
                [traj[w, 1], traj[w + 1, 1]],
                [traj[w, 0], traj[w + 1, 0]],
                color=colors[w], linewidth=1.5,
            )
        ax.scatter(traj[0, 1],  traj[0, 0],  color="lime",   s=50, zorder=5)
        ax.scatter(traj[-1, 1], traj[-1, 0], color="red",    s=50, zorder=5)
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(GRID_SIZE - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{i + 1}", fontsize=9)
        ax.set_aspect("equal")

    for j in range(len(trajs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[vis] → {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Expert vs Generated comparison
# ---------------------------------------------------------------------------

def plot_expert_vs_generated(
    expert_norm: np.ndarray,
    gen_norm: torch.Tensor,
    out_path: str,
    n: int = 6,
):
    expert = _denorm(expert_norm[:n])
    gen    = _denorm(gen_norm.cpu().numpy()[:n])

    fig, axes = plt.subplots(2, n, figsize=(n * 2.8, 5.5))

    for i in range(n):
        for row, traj, label, color in [
            (0, expert[i], "Expert (Q-agent)", "steelblue"),
            (1, gen[i],    "Diffusion Generated",   "darkorange"),
        ]:
            ax = axes[row, i]
            ax.plot(traj[:, 1], traj[:, 0], "-o",
                    color=color, markersize=1.5, linewidth=1.2)
            ax.scatter(traj[0, 1],  traj[0, 0],  color="lime", s=40, zorder=5)
            ax.scatter(traj[-1, 1], traj[-1, 0], color="red",  s=40, zorder=5)
            ax.set_xlim(-0.5, GRID_SIZE - 0.5)
            ax.set_ylim(GRID_SIZE - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if i == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")

    fig.suptitle("Expert Trajectories vs Diffusion Generated", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[vis] → {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Denoising chain for one trajectory
# ---------------------------------------------------------------------------

def plot_denoising_chain(
    chain: list[torch.Tensor],
    out_path: str,
    n_frames: int = 8,
):
    """Show how one trajectory evolves from pure noise to clean path."""
    step_indices = np.linspace(0, len(chain) - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2.8, 3))

    for col, step_idx in enumerate(step_indices):
        traj_norm = chain[step_idx][0].cpu().numpy()   # first sample (H, 2)
        traj = _denorm(traj_norm)
        ax = axes[col]
        ax.plot(traj[:, 1], traj[:, 0], "b-", linewidth=1.2)
        ax.scatter(traj[0, 1],  traj[0, 0],  color="lime", s=30, zorder=5)
        ax.scatter(traj[-1, 1], traj[-1, 0], color="red",  s=30, zorder=5)
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(GRID_SIZE - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        t_label = len(chain) - 1 - step_idx  # t goes from T→0
        ax.set_title(f"t={t_label}", fontsize=9)

    fig.suptitle("Denoising Chain: τ_T (noise) → τ_0 (trajectory)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[vis] → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize(
    n_samples: int = 16,
    device: str = None,
    checkpoint_dir: str = None,
    data_dir: str = None,
    results_dir: str = None,
    ckpt_name: str = "diffusion_best.pt",
    out_prefix: str = "diffusion",
    *,
    plot_samples_grid: bool = False,
    plot_expert_comparison: bool = False,
    plot_denoising: bool = False,
):
    checkpoint_dir = checkpoint_dir or os.path.join(_ROOT, "checkpoints")
    data_dir       = data_dir       or os.path.join(_ROOT, "data")
    results_dir    = results_dir    or os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        sys.exit(
            f"[ERROR] {ckpt_path} not found.\n"
            "Run Phase 2 training first:  python run_phase2.py"
        )

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
                 "cuda" if torch.cuda.is_available() else "cpu")

    unet, diffusion, cfg = _load_model(ckpt_path, device)
    H, D = cfg["horizon"], cfg["state_dim"]
    context_dim = cfg.get("context_dim", 0)

    # Goal-conditioned sampling: diverse starts, fixed goal (10, 10)
    g_norm = goal_normalized()
    goal_t = torch.tensor(g_norm, device=device, dtype=torch.float32).unsqueeze(0).repeat(n_samples, 1)
    starts_list = []
    for i in range(n_samples):
        r = float(np.random.randint(0, GRID_SIZE))
        c = float(np.random.randint(0, GRID_SIZE))
        starts_list.append(normalize_grid([r, c]))
    starts = torch.tensor(np.stack(starts_list), device=device, dtype=torch.float32)

    print(f"[vis] Sampling {n_samples} trajectories (context_dim={context_dim}) …")
    sample_kw = dict(
        batch_size=n_samples,
        horizon=H,
        state_dim=D,
        start=starts,
        goal=goal_t,
    )
    if context_dim > 0:
        sample_kw["context"] = _build_sample_context(starts, goal_t, context_dim, device)
    samples = diffusion.sample(**sample_kw)

    if plot_samples_grid:
        plot_samples_grid(
            samples,
            out_path=os.path.join(results_dir, f"{out_prefix}_samples_grid.png"),
            n=n_samples,
            title=f"Diffusion Generated Trajectories ({ckpt_name})",
        )

    if plot_expert_comparison:
        train_path = os.path.join(data_dir, "trajectories_taskcond_train.npy")
        if not os.path.exists(train_path):
            train_path = os.path.join(data_dir, "trajectories_train.npy")
        if os.path.exists(train_path):
            expert = np.load(train_path).astype(np.float32)
            np.random.shuffle(expert)
            plot_expert_vs_generated(
                expert,
                samples,
                out_path=os.path.join(results_dir, f"{out_prefix}_vs_expert.png"),
            )

    if plot_denoising:
        print("[vis] Generating denoising chain …")
        chain_kw = dict(batch_size=1, horizon=H, state_dim=D, return_chain=True)
        if context_dim > 0:
            s0 = starts[:1]
            g0 = goal_t[:1]
            chain_kw.update(
                start=s0,
                goal=g0,
                context=_build_sample_context(s0, g0, context_dim, device),
            )
        _, chain = diffusion.sample(**chain_kw)
        plot_denoising_chain(
            chain,
            out_path=os.path.join(results_dir, f"{out_prefix}_denoising_chain.png"),
        )

    print("[vis] Visualisation complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualise diffusion generated trajectories")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--device",    type=str, default=None)
    p.add_argument("--ckpt",      type=str, default="diffusion_best.pt")
    p.add_argument("--out_prefix", type=str, default=None,
                   help="Output filename prefix (default: diffusion or diffusion_taskcond)")
    args = p.parse_args()
    prefix = args.out_prefix or (
        "diffusion_taskcond" if args.ckpt == "diffusion_taskcond.pt" else "diffusion"
    )
    visualize(
        n_samples=args.n_samples,
        device=args.device,
        ckpt_name=args.ckpt,
        out_prefix=prefix,
    )
