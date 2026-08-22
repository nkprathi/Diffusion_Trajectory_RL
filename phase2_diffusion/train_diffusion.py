"""
train_diffusion.py  —  Phase 2

Train the Temporal U-Net diffusion model on expert trajectories prepared in Phase 1.

Inputs  (from data/):
    trajectories_train.npy  — (N_train, H, 2) normalised float32 trajectories
    trajectories_val.npy    — (N_val,   H, 2) normalised float32 trajectories

Outputs (to checkpoints/ and results/):
    checkpoints/diffusion_best.pt   — best model weights (lowest val loss)
    checkpoints/diffusion_final.pt  — final model weights
    results/loss_curve.png     — train / val loss over epochs
    results/diffusion_samples.png   — 8 sampled trajectories after training

Usage:
    python phase2_diffusion/train_diffusion.py                  # defaults
    python phase2_diffusion/train_diffusion.py --epochs 300 --batch_size 64

References:
    Ho et al. (2020) — DDPM
    Janner et al. (2022) — Diffuser
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- path setup: allow running from project root or this folder -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from phase2_diffusion.noise_schedule import NoiseSchedule
from phase2_diffusion.unet import TemporalUNet
from phase2_diffusion.diffusion import GaussianDiffusion
from phase2_diffusion.trajectory_utils import (
    CONTEXT_DIM,
    make_context_from_batch,
    goal_normalized,
    normalize_grid,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except ImportError:
    _PLT = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(data_dir: str):
    train_path = os.path.join(data_dir, "trajectories_train.npy")
    val_path   = os.path.join(data_dir, "trajectories_val.npy")

    if not os.path.exists(train_path):
        sys.exit(
            f"[ERROR] {train_path} not found.\n"
            "Run Phase 1 first:  python run_phase1.py"
        )

    train_data = np.load(train_path).astype(np.float32)   # (N, H, 2)
    val_data   = np.load(val_path).astype(np.float32)

    print(f"[data] train: {train_data.shape}   val: {val_data.shape}")
    return train_data, val_data


def _plot_loss(
    train_losses: list[float],
    val_losses: list[float],
    out_path: str,
):
    if not _PLT:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_losses, label="Train loss", linewidth=1.5)
    ax.plot(val_losses,   label="Val loss",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Diffusion Training — Noise-Prediction Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Loss curve saved → {out_path}")


def _plot_samples(
    trajectories: torch.Tensor,
    out_path: str,
    grid_size: int = 12,
):
    """Plot generated trajectories in grid coordinates."""
    if not _PLT:
        return

    # Denormalise: [-1, 1] → [0, grid_size-1]
    trajs = trajectories.cpu().numpy()
    trajs = (trajs + 1.0) / 2.0 * (grid_size - 1)

    n = min(8, trajs.shape[0])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        traj = trajs[i]           # (H, 2)
        ax.plot(traj[:, 1], traj[:, 0], "b-o", markersize=2, linewidth=1.2)
        ax.scatter(traj[0, 1],  traj[0, 0],  color="green", s=60, zorder=5, label="start")
        ax.scatter(traj[-1, 1], traj[-1, 0], color="red",   s=60, zorder=5, label="goal")
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_title(f"Sample {i + 1}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Diffusion Generated Trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Sample trajectories saved → {out_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str = None,
    checkpoint_dir: str = None,
    results_dir: str = None,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 2e-4,
    T: int = 100,
    schedule: str = "cosine",
    horizon: int = 16,
    state_dim: int = 2,
    channels: tuple = (32, 64, 128, 256),
    time_embed_dim: int = 128,
    num_workers: int = 0,
    device: str = None,
    log_every: int = 10,
):
    # --- resolve directories -----------------------------------------------
    data_dir       = data_dir       or os.path.join(_ROOT, "data")
    checkpoint_dir = checkpoint_dir or os.path.join(_ROOT, "checkpoints")
    results_dir    = results_dir    or os.path.join(_ROOT, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    # --- device ------------------------------------------------------------
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
                 "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # --- data --------------------------------------------------------------
    train_data, val_data = _load_data(data_dir)
    train_ds = TensorDataset(torch.from_numpy(train_data))
    val_ds   = TensorDataset(torch.from_numpy(val_data))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- model + schedule --------------------------------------------------
    noise_sched = NoiseSchedule(T=T, schedule=schedule, device=device)
    unet = TemporalUNet(
        state_dim=state_dim,
        horizon=horizon,
        channels=channels,
        time_embed_dim=time_embed_dim,
        context_dim=CONTEXT_DIM,
    )
    diffusion = GaussianDiffusion(unet, noise_sched, device=device)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"[model] TemporalUNet params: {n_params:,}")

    optimiser = torch.optim.AdamW(unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    # --- training ----------------------------------------------------------
    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val = float("inf")

    print(f"\n[train] Starting Diffusion training for {epochs} epochs …\n")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ---- Train --------------------------------------------------------
        unet.train()
        epoch_train = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            context = make_context_from_batch(batch)
            loss = diffusion.training_loss(batch, context=context)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimiser.step()
            epoch_train += loss.item()
        epoch_train /= len(train_loader)
        train_losses.append(epoch_train)

        # ---- Validation ---------------------------------------------------
        unet.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                context = make_context_from_batch(batch)
                epoch_val += diffusion.training_loss(batch, context=context).item()
        epoch_val /= len(val_loader)
        val_losses.append(epoch_val)

        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"train={epoch_train:.5f}  val={epoch_val:.5f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"elapsed={elapsed:.0f}s"
            )

        # ---- Save best ----------------------------------------------------
        if epoch_val < best_val:
            best_val = epoch_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": unet.state_dict(),
                    "val_loss": best_val,
                    "config": {
                        "state_dim": state_dim,
                        "horizon": horizon,
                        "channels": channels,
                        "time_embed_dim": time_embed_dim,
                        "T": T,
                        "schedule": schedule,
                        "context_dim": CONTEXT_DIM,
                        "goal_conditioned": True,
                    },
                },
                os.path.join(checkpoint_dir, "diffusion_best.pt"),
            )

    # ---- Save final -------------------------------------------------------
    torch.save(
        {
            "epoch": epochs,
            "model_state": unet.state_dict(),
            "val_loss": val_losses[-1],
            "config": {
                "state_dim": state_dim,
                "horizon": horizon,
                "channels": channels,
                "time_embed_dim": time_embed_dim,
                "T": T,
                "schedule": schedule,
                "context_dim": CONTEXT_DIM,
                "goal_conditioned": True,
            },
        },
        os.path.join(checkpoint_dir, "diffusion_final.pt"),
    )

    total_time = time.time() - t0
    print(f"\n[train] Done — {total_time:.0f}s total. Best val loss: {best_val:.5f}")
    print(f"[train] Checkpoints → {checkpoint_dir}")

    # ---- Plots ------------------------------------------------------------
    _plot_loss(
        train_losses, val_losses,
        os.path.join(results_dir, "loss_curve.png"),
    )

    # Quick sample with best model
    print("[sample] Generating 8 sample trajectories …")
    ckpt = torch.load(
        os.path.join(checkpoint_dir, "diffusion_best.pt"),
        map_location=device,
        weights_only=True,
    )
    unet.load_state_dict(ckpt["model_state"])
    unet.eval()
    # Goal-conditioned samples: random starts, fixed goal (10, 10)
    g_norm = goal_normalized()
    goal_t = torch.tensor(g_norm, device=device, dtype=torch.float32).unsqueeze(0).repeat(8, 1)
    starts = torch.from_numpy(np.stack([
        normalize_grid([0, 0]), normalize_grid([5, 2]), normalize_grid([2, 8]),
        normalize_grid([8, 1]), normalize_grid([1, 5]), normalize_grid([6, 6]),
        normalize_grid([3, 3]), normalize_grid([9, 4]),
    ])).to(device)
    context = torch.cat([starts, goal_t], dim=-1)
    samples = diffusion.sample(
        batch_size=8,
        horizon=horizon,
        state_dim=state_dim,
        start=starts,
        goal=goal_t,
        context=context,
    )
    _plot_samples(samples, os.path.join(results_dir, "diffusion_samples.png"))

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train diffusion for trajectory planning")
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--T",              type=int,   default=100,
                   help="Number of diffusion timesteps")
    p.add_argument("--schedule",       type=str,   default="cosine",
                   choices=["linear", "cosine", "exponential"])
    p.add_argument("--horizon",        type=int,   default=16,
                   help="Waypoints per trajectory (must match Phase 1)")
    p.add_argument("--time_embed_dim", type=int,   default=128)
    p.add_argument("--log_every",      type=int,   default=10)
    p.add_argument("--device",         type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        schedule=args.schedule,
        horizon=args.horizon,
        time_embed_dim=args.time_embed_dim,
        log_every=args.log_every,
        device=args.device,
    )
