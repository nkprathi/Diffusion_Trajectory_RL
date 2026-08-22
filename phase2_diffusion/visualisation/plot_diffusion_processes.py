"""
plot_diffusion_processes.py  (v3)

Two publication figures:
  results/forward_process.png  — real expert trajectory corrupted t=0 → T
  results/reverse_process.png  — Diffusion denoising chain  t=T → t=0

Called automatically from run_phase2.py after Diffusion training / path visualisation.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from phase2_diffusion.diffusion import GaussianDiffusion
from phase2_diffusion.noise_schedule import NoiseSchedule
from phase2_diffusion.trajectory_utils import make_context
from phase2_diffusion.unet import TemporalUNet

GRID_SIZE = 12
DEFAULT_T = 100
DEFAULT_SCHEDULE = "cosine"
DEFAULT_HORIZON = 16


def to_grid(norm_xy, grid_size: int = GRID_SIZE):
    """Normalised (…,2) → grid coords clamped to [0, grid_size-1]."""
    return np.clip(
        (np.asarray(norm_xy) + 1.0) / 2.0 * (grid_size - 1),
        0,
        grid_size - 1,
    )


def grid_bg(ax, grid_size: int = GRID_SIZE, facecolor="#F0F4FF"):
    for r in range(grid_size):
        for c in range(grid_size):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (c, grid_size - 1 - r),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    facecolor=facecolor,
                    edgecolor="#CCCCCC",
                    linewidth=0.4,
                )
            )
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_traj(ax, traj_norm, color, lw, ms, alpha, start_norm, goal_norm, grid_size=GRID_SIZE):
    g = to_grid(traj_norm, grid_size).copy()
    g[0] = to_grid(start_norm, grid_size)
    g[-1] = to_grid(goal_norm, grid_size)

    cols = g[:, 1] + 0.5
    rows = (grid_size - 1 - g[:, 0]) + 0.5

    ax.plot(cols, rows, "-", color=color, lw=lw, alpha=alpha, zorder=3)
    ax.scatter(cols[1:-1], rows[1:-1], s=ms, color=color, alpha=alpha, zorder=4)
    ax.scatter(
        cols[0], rows[0], s=110, color="#2ECC71",
        edgecolors="white", lw=1.4, zorder=6,
    )
    ax.scatter(
        cols[-1], rows[-1], s=140, color="#E74C3C",
        edgecolors="white", lw=1.4, marker="*", zorder=6,
    )


def add_arrows(fig, n_panels):
    for i in range(n_panels - 1):
        x = (i + 1) / n_panels
        fig.text(
            x, 0.225, "→", fontsize=13, color="#999999",
            ha="center", va="center", transform=fig.transFigure,
        )


def _select_trajectory(
    train_data: np.ndarray,
    *,
    traj_index: int | None = None,
    start_pos: tuple[int, int] | None = None,
) -> tuple[int, torch.Tensor]:
    """Pick expert trajectory by index or nearest start cell."""
    if traj_index is not None:
        idx = int(traj_index)
        if idx < 0 or idx >= len(train_data):
            raise IndexError(f"traj_index {idx} out of range [0, {len(train_data)})")
        traj = torch.tensor(train_data[idx], dtype=torch.float32).unsqueeze(0)
        return idx, traj

    if start_pos is not None:
        target = np.array(start_pos, dtype=np.float32)
        target_n = (2.0 * target / (GRID_SIZE - 1) - 1.0)
        starts = train_data[:, 0, :]
        dist = np.linalg.norm(starts - target_n, axis=1)
        idx = int(np.argmin(dist))
        traj = torch.tensor(train_data[idx], dtype=torch.float32).unsqueeze(0)
        return idx, traj

    goal_n = np.array([0.818, 0.818])
    idx = int(np.argmin(np.linalg.norm(train_data[:, -1] - goal_n, axis=1)))
    traj = torch.tensor(train_data[idx], dtype=torch.float32).unsqueeze(0)
    return idx, traj


def plot_diffusion_processes(
    *,
    device: str | None = None,
    results_dir: str | None = None,
    checkpoint: str | None = None,
    train_path: str | None = None,
    T: int = DEFAULT_T,
    schedule: str = DEFAULT_SCHEDULE,
    horizon: int = DEFAULT_HORIZON,
    traj_index: int | None = None,
    start_pos: tuple[int, int] | None = (0, 2),
) -> dict[str, str]:
    """
    Regenerate forward_process.png and reverse_process.png from the latest Diffusion checkpoint.

    Returns dict with keys ``forward`` and ``reverse`` pointing to saved PNG paths.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = results_dir or os.path.join(_ROOT, "results")
    checkpoint = checkpoint or os.path.join(_ROOT, "checkpoints", "diffusion_best.pt")
    train_path = train_path or os.path.join(_ROOT, "data", "trajectories_train.npy")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint}")
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training trajectories not found: {train_path}")

    sched = NoiseSchedule(T=T, schedule=schedule, device=device)

    train_data = np.load(train_path)
    best_idx, clean_traj = _select_trajectory(
        train_data, traj_index=traj_index, start_pos=start_pos,
    )
    start_grid = ((clean_traj[0, 0].numpy() + 1) / 2 * 11).round()
    end_grid = ((clean_traj[0, -1].numpy() + 1) / 2 * 11).round()
    print(
        f"Expert traj #{best_idx}  start={start_grid.tolist()}  end={end_grid.tolist()}"
        + (f"  (requested start={start_pos})" if start_pos else "")
    )

    # ── Forward process ─────────────────────────────────────────────────────
    n_fwd = 7
    t_fwd = [0, 10, 20, 40, 55, 80, 99]
    labels_fwd = [
        "t = 0\n(clean)", "t = 10", "t = 20",
        "t = 40", "t = 55", "t = 80", "t = 100\n(pure noise)",
    ]

    torch.manual_seed(42)
    noise_seed = torch.randn(1, horizon, 2)
    colours_fwd = plt.cm.plasma(np.linspace(0.08, 0.92, n_fwd))

    fig1 = plt.figure(figsize=(20, 8))
    fig1.patch.set_facecolor("white")
    gs1 = gridspec.GridSpec(2, n_fwd, figure=fig1, hspace=0.50, wspace=0.18)

    ax_s = fig1.add_subplot(gs1[0, :])
    ts = np.arange(T)
    ax_s.plot(
        ts, sched.sqrt_alpha_bars.cpu().numpy(), color="#4C72B0", lw=2.4,
        label=r"$\sqrt{\bar\alpha_t}$  — signal coefficient",
    )
    ax_s.plot(
        ts, sched.sqrt_1m_ab.cpu().numpy(), color="#E74C3C", lw=2.4,
        label=r"$\sqrt{1-\bar\alpha_t}$  — noise coefficient",
    )
    ax_s.plot(
        ts, sched.alpha_bars.cpu().numpy(), color="#2ECC71", lw=1.6, ls="--",
        label=r"$\bar\alpha_t$  — cumulative signal",
    )
    for t_v, col in zip(t_fwd, colours_fwd):
        ax_s.axvline(t_v, color=col, lw=1.2, ls=":", alpha=0.85)
        ax_s.text(
            t_v + 0.8, 0.04, f"t={t_v if t_v < 99 else 100}",
            fontsize=7, color=col, rotation=90, va="bottom",
        )
    ax_s.set_xlim(0, T - 1)
    ax_s.set_ylim(-0.02, 1.08)
    ax_s.set_xlabel("Diffusion timestep  t", fontsize=10)
    ax_s.set_ylabel("Coefficient value", fontsize=10)
    ax_s.set_title(
        "Cosine Noise Schedule (T = 100)  —  Signal decays, Noise grows",
        fontsize=11, fontweight="bold",
    )
    ax_s.legend(fontsize=9, loc="center right")
    ax_s.grid(alpha=0.3, axis="y")
    ax_s.spines[["top", "right"]].set_visible(False)

    start_np = clean_traj[0, 0].numpy()
    goal_np = clean_traj[0, -1].numpy()
    for idx, (t_v, col, lbl) in enumerate(zip(t_fwd, colours_fwd, labels_fwd)):
        ax = fig1.add_subplot(gs1[1, idx])
        grid_bg(ax)
        if t_v == 0:
            tau_t = clean_traj[0].numpy()
        else:
            tau_t = sched.q_sample(
                clean_traj, torch.tensor([t_v]), noise_seed,
            )[0].numpy()
        frac = t_v / T
        lw = max(0.7, 2.4 * (1 - frac))
        ms = max(6, 40 * (1 - frac))
        alpha = max(0.30, 1.0 - 0.65 * frac)
        plot_traj(ax, tau_t, col, lw, ms, alpha, start_np, goal_np)
        ax.set_title(lbl, fontsize=9, fontweight="bold", pad=5, color=col)
        if t_v == 0:
            ann = r"$\tau_0$  (expert)"
        else:
            ab_v = float(sched.alpha_bars[min(t_v, T - 1)].item())
            ann = rf"$\bar\alpha_t = {ab_v:.2f}$"
        ax.text(
            0.5, -0.10, ann, transform=ax.transAxes, fontsize=6.5, ha="center",
            bbox=dict(
                boxstyle="round,pad=0.22", facecolor="#FFFDE7",
                edgecolor="#CCBB00", alpha=0.9,
            ),
        )

    add_arrows(fig1, n_fwd)
    fig1.suptitle(
        "Forward Diffusion Process:  Clean Expert Trajectory  →  Pure Gaussian Noise",
        fontsize=13, fontweight="bold", y=1.01,
    )
    forward_path = os.path.join(results_dir, "forward_process.png")
    fig1.savefig(forward_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"Saved → {forward_path}")

    # ── Reverse process ─────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    unet = TemporalUNet(
        state_dim=cfg["state_dim"],
        horizon=cfg["horizon"],
        channels=tuple(cfg["channels"]),
        time_embed_dim=cfg["time_embed_dim"],
        context_dim=cfg.get("context_dim", 4),
    )
    unet.load_state_dict(ckpt["model_state"])
    unet.eval()
    diff = GaussianDiffusion(unet, sched, device=device)

    start_t = clean_traj[0, 0].unsqueeze(0)
    goal_t = clean_traj[0, -1].unsqueeze(0)
    ctx = make_context(start_t, goal_t)

    torch.manual_seed(0)
    with torch.no_grad():
        _, chain = diff.sample(
            1, cfg["horizon"], cfg["state_dim"],
            start=start_t, goal=goal_t,
            context=ctx, return_chain=True,
        )

    t_rev = [100, 80, 55, 40, 20, 10, 0]
    chain_idx = [T - t for t in t_rev]
    labels_rev = [
        "t = 100\n(pure noise)", "t = 80", "t = 55",
        "t = 40", "t = 20", "t = 10", "t = 0\n(generated)",
    ]
    n_rev = len(t_rev)
    colours_rev = plt.cm.viridis(np.linspace(0.85, 0.1, n_rev))

    fig2 = plt.figure(figsize=(20, 8))
    fig2.patch.set_facecolor("white")
    gs2 = gridspec.GridSpec(2, n_rev, figure=fig2, hspace=0.50, wspace=0.18)

    ax_v = fig2.add_subplot(gs2[0, :])
    post_std = np.sqrt(np.clip(sched.posterior_variance.cpu().numpy(), 0, None))
    ax_v.fill_between(np.arange(T), post_std, alpha=0.2, color="#7B68EE")
    ax_v.plot(
        np.arange(T), post_std, color="#7B68EE", lw=2.2,
        label=r"$\sigma_t = \sqrt{\tilde\beta_t}$  (noise added per reverse step)",
    )
    ax_v.plot(
        np.arange(T), sched.sqrt_1m_ab.cpu().numpy(), color="#E74C3C",
        lw=1.8, ls="--",
        label=r"$\sqrt{1-\bar\alpha_t}$  (overall noise level)",
    )
    for t_v, col in zip(t_rev, colours_rev):
        if 0 < t_v < T:
            ax_v.axvline(t_v, color=col, lw=1.2, ls=":", alpha=0.8)
            ax_v.text(
                t_v - 1.5, 0.005, f"t={t_v}",
                fontsize=7, color=col, rotation=90, va="bottom",
            )
    # Flip x-axis: t=T (pure noise) on the left, t=0 (clean) on the right
    # so the generation direction reads naturally left → right.
    ax_v.set_xlim(T - 1, 0)
    ax_v.set_xlabel(
        "Diffusion timestep  t          "
        "[pure noise  t = T  ←  axis  →  t = 0  clean trajectory]",
        fontsize=10,
    )
    ax_v.set_ylabel("Standard deviation", fontsize=10)
    ax_v.set_title(
        r"Reverse Posterior  $p_\theta(\tau_{t-1}|\tau_t)$"
        r"  —  both $\sigma_t$ and $\sqrt{1-\bar\alpha_t}$ decrease as generation progresses"
        r"  (left $\to$ right)",
        fontsize=11, fontweight="bold",
    )
    ax_v.legend(fontsize=9, loc="upper right")
    ax_v.grid(alpha=0.3, axis="y")
    ax_v.spines[["top", "right"]].set_visible(False)

    start_np = start_t[0].numpy()
    goal_np = goal_t[0].numpy()
    for idx, (t_v, ci, col, lbl) in enumerate(
        zip(t_rev, chain_idx, colours_rev, labels_rev)
    ):
        clarity = 1.0 - t_v / T
        bg = "#F5F0FF" if t_v > 50 else "#EAF7EA"
        ax = fig2.add_subplot(gs2[1, idx])
        grid_bg(ax, facecolor=bg)
        traj_np = chain[ci][0].numpy()
        lw = 0.7 + 2.4 * clarity
        ms = 6 + 44 * clarity
        alpha = 0.30 + 0.70 * clarity
        plot_traj(ax, traj_np, col, lw, ms, alpha, start_np, goal_np)
        ax.set_title(lbl, fontsize=9, fontweight="bold", pad=5, color=col)
        if t_v == T:
            ann = r"$\tau_T \sim \mathcal{N}(0,I)$"
        elif t_v == 0:
            ann = r"$\tau_0$  (start→goal path)"
        else:
            ann = (
                r"$\mu_t = \frac{1}{\sqrt{\alpha_t}}"
                r"(\tau_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\varepsilon)$"
            )
        ax.text(
            0.5, -0.10, ann, transform=ax.transAxes, fontsize=5.8, ha="center",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="#F0FFF4" if clarity > 0.5 else "#FFF0F0",
                edgecolor=col, alpha=0.92,
            ),
        )

    add_arrows(fig2, n_rev)
    fig2.suptitle(
        "Reverse Diffusion Process:  Pure Gaussian Noise  →  Generated Waypoints  (actual Diffusion chain)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    reverse_path = os.path.join(results_dir, "reverse_process.png")
    fig2.savefig(reverse_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved → {reverse_path}")

    return {"forward": forward_path, "reverse": reverse_path}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot forward/reverse diffusion process figures")
    p.add_argument("--start", type=int, nargs=2, default=[0, 2], metavar=("ROW", "COL"),
                   help="Pick expert traj nearest this start (default: 0 2)")
    p.add_argument("--traj_index", type=int, default=None,
                   help="Override: use this training trajectory index")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    plot_diffusion_processes(
        device=args.device,
        start_pos=None if args.traj_index is not None else tuple(args.start),
        traj_index=args.traj_index,
    )
