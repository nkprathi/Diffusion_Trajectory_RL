"""
train_diffusion_taskcond.py — Phase 3 task-conditioned diffusion (6-dim context)

Trains on trajectories_taskcond_*.npy + context_taskcond_*.npy
Saves checkpoints/diffusion_taskcond.pt (does not overwrite diffusion_best.pt).

Usage:
    python phase2_diffusion/train_diffusion_taskcond.py --epochs 120
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from phase2_diffusion.diffusion import GaussianDiffusion
from phase2_diffusion.noise_schedule import NoiseSchedule
from phase2_diffusion.trajectory_utils import TASK_CONTEXT_DIM, goal_normalized, normalize_grid
from phase2_diffusion.unet import TemporalUNet

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except ImportError:
    _PLT = False

OUT_CKPT = "diffusion_taskcond.pt"


def _load_taskcond_data(data_dir: str):
    train_traj = os.path.join(data_dir, "trajectories_taskcond_train.npy")
    val_traj = os.path.join(data_dir, "trajectories_taskcond_val.npy")
    train_ctx = os.path.join(data_dir, "context_taskcond_train.npy")
    val_ctx = os.path.join(data_dir, "context_taskcond_val.npy")
    for p in (train_traj, val_traj, train_ctx, val_ctx):
        if not os.path.isfile(p):
            sys.exit(
                f"Missing {p}\n"
                "Run prepare_taskcond_dataset.py first."
            )
    tr_train = np.load(train_traj).astype(np.float32)
    tr_val = np.load(val_traj).astype(np.float32)
    cx_train = np.load(train_ctx).astype(np.float32)
    cx_val = np.load(val_ctx).astype(np.float32)
    print(f"[data] train traj {tr_train.shape}  ctx {cx_train.shape}")
    print(f"[data] val   traj {tr_val.shape}  ctx {cx_val.shape}")
    return tr_train, tr_val, cx_train, cx_val


def _epoch_validation_loss(
    diffusion: GaussianDiffusion,
    val_loader: DataLoader,
    device: str,
    epoch: int,
    mc_samples: int,
) -> float:
    """Mean batch validation loss with fixed RNG per (epoch, batch)."""
    total = 0.0
    for batch_idx, (batch, context) in enumerate(val_loader):
        batch = batch.to(device)
        context = context.to(device)
        total += diffusion.validation_loss(
            batch,
            context,
            seed=epoch * 10_000 + batch_idx,
            mc_samples=mc_samples,
        )
    return total / len(val_loader)


def _plot_loss(train_losses, val_losses, out_path: str, smooth_window: int = 5):
    if not _PLT:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val (MC avg)", linestyle="--", linewidth=1.5)
    if smooth_window > 1 and len(val_losses) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(val_losses, kernel, mode="valid")
        smooth_x = epochs[smooth_window - 1:]
        ax.plot(
            smooth_x,
            smoothed,
            label=f"Val smooth ({smooth_window} ep)",
            linewidth=2.0,
            color="#E4572E",
            alpha=0.85,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Phase 3 — Task-Conditioned Diffusion (start + goal + laundry)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {out_path}")


def train_taskcond(
    *,
    data_dir: str | None = None,
    checkpoint_dir: str | None = None,
    results_dir: str | None = None,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 1e-4,
    T: int = 100,
    schedule: str = "cosine",
    horizon: int = 16,
    device: str | None = None,
    log_every: int = 10,
    init_ckpt: str | None = None,
    val_mc_samples: int = 8,
) -> str:
    data_dir = data_dir or os.path.join(_ROOT, "data")
    checkpoint_dir = checkpoint_dir or os.path.join(_ROOT, "checkpoints")
    results_dir = results_dir or os.path.join(_ROOT, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    print(f"[train] Device: {device}")

    tr_train, tr_val, cx_train, cx_val = _load_taskcond_data(data_dir)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(tr_train), torch.from_numpy(cx_train)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(tr_val), torch.from_numpy(cx_val)),
        batch_size=batch_size, shuffle=False,
    )

    channels = (32, 64, 128, 256)
    time_embed_dim = 128
    noise_sched = NoiseSchedule(T=T, schedule=schedule, device=device)
    unet = TemporalUNet(
        state_dim=2,
        horizon=horizon,
        channels=channels,
        time_embed_dim=time_embed_dim,
        context_dim=TASK_CONTEXT_DIM,
    )

    if init_ckpt and os.path.isfile(init_ckpt):
        ckpt = torch.load(init_ckpt, map_location=device, weights_only=True)
        old = ckpt["model_state"]
        old_ctx = ckpt.get("config", {}).get("context_dim", 0)
        if old_ctx != TASK_CONTEXT_DIM:
            old = {
                k: v for k, v in old.items()
                if not k.startswith("context_proj.")
            }
        missing, unexpected = unet.load_state_dict(old, strict=False)
        print(
            f"[train] Partial init from {init_ckpt}  "
            f"missing={len(missing)}  unexpected={len(unexpected)}"
        )

    diffusion = GaussianDiffusion(unet, noise_sched, device=device)
    optimiser = torch.optim.AdamW(unet.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    train_losses, val_losses = [], []
    best_val = float("inf")
    out_path = os.path.join(checkpoint_dir, OUT_CKPT)
    t0 = time.time()

    print(f"\n[train] Task-conditioned diffusion — {epochs} epochs")
    print(f"[train] Val loss: {val_mc_samples} MC samples/batch, fixed seed per epoch\n")
    for epoch in range(1, epochs + 1):
        unet.train()
        ep_train = 0.0
        for batch, context in train_loader:
            batch = batch.to(device)
            context = context.to(device)
            loss = diffusion.training_loss(batch, context=context)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimiser.step()
            ep_train += loss.item()
        ep_train /= len(train_loader)
        train_losses.append(ep_train)

        unet.eval()
        with torch.no_grad():
            ep_val = _epoch_validation_loss(
                diffusion, val_loader, device, epoch, val_mc_samples
            )
        val_losses.append(ep_val)
        sched.step()

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}  train={ep_train:.5f}  val={ep_val:.5f}  "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )

        if ep_val < best_val:
            best_val = ep_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": unet.state_dict(),
                    "val_loss": best_val,
                    "config": {
                        "state_dim": 2,
                        "horizon": horizon,
                        "channels": channels,
                        "time_embed_dim": time_embed_dim,
                        "T": T,
                        "schedule": schedule,
                        "context_dim": TASK_CONTEXT_DIM,
                        "goal_conditioned": True,
                        "task_conditioned": True,
                        "data_source": "astar+static+dynamic_pet",
                    },
                },
                out_path,
            )

    print(f"\n[train] Done in {time.time() - t0:.0f}s. Best val={best_val:.5f}")
    print(f"[train] Saved → {out_path}")
    _plot_loss(train_losses, val_losses, os.path.join(results_dir, "diffusion_taskcond_loss_curve.png"))
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train task-conditioned diffusion (Phase 3)")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--init_ckpt", type=str, default=None,
                     help="Optional diffusion_best.pt for partial weight init")
    p.add_argument("--val_mc_samples", type=int, default=8,
                     help="MC noise/t draws per val batch (lower variance curve)")
    args = p.parse_args()
    init = args.init_ckpt or os.path.join(_ROOT, "checkpoints", "diffusion_best.pt")
    if not os.path.isfile(init):
        init = None
    train_taskcond(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        init_ckpt=init,
        val_mc_samples=args.val_mc_samples,
    )
