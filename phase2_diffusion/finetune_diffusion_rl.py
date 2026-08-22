"""
finetune_diffusion_rl.py  —  Phase 2.5

Merge A* expert trajectories with successful PPO rollouts, then fine-tune
the Phase 2 Diffusion checkpoint for dynamic task-aware planning.

Inputs:
    data/raw_trajectories.npy   — Phase 1 A* experts
    data/rl_trajectories.npy    — Phase 3 successful PPO rollouts

Outputs:
    data/trajectories_rl_finetune_train.npy
    data/trajectories_rl_finetune_val.npy
    checkpoints/diffusion_rl_finetuned.pt
    results/phase2_5_loss_curve.png
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

from phase1_home_navigation.prepare_trajectory_dataset import (
    GRID_SIZE,
    normalize,
    resample_path,
)
from phase2_diffusion.diffusion import GaussianDiffusion
from phase2_diffusion.noise_schedule import NoiseSchedule
from phase2_diffusion.trajectory_utils import CONTEXT_DIM, make_context_from_batch
from phase2_diffusion.unet import TemporalUNet

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except ImportError:
    _PLT = False

TRAIN_OUT = "trajectories_rl_finetune_train.npy"
VAL_OUT = "trajectories_rl_finetune_val.npy"
FINETUNED_CKPT = "diffusion_rl_finetuned.pt"


def process_raw_paths(raw: np.ndarray, horizon: int) -> np.ndarray:
    """Variable-length grid paths → (N, H, 2) normalised float32."""
    processed = []
    skipped = 0
    for traj in raw:
        path = np.array(traj, dtype=np.float32)
        if path.shape[0] < 2:
            skipped += 1
            continue
        path_norm = normalize(path, GRID_SIZE)
        processed.append(resample_path(path_norm, horizon))
    if skipped:
        print(f"  skipped {skipped} degenerate paths (< 2 steps)")
    return np.array(processed, dtype=np.float32)


def build_merged_dataset(
    *,
    data_dir: str | None = None,
    expert_path: str | None = None,
    rl_path: str | None = None,
    horizon: int = 16,
    val_split: float = 0.1,
    seed: int = 42,
    include_all_rl: bool = True,
    max_expert: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load A* + RL raw paths, resample to H waypoints, split train/val.

    Returns (train_data, val_data, stats_dict).
    """
    data_dir = data_dir or os.path.join(_ROOT, "data")
    expert_path = expert_path or os.path.join(data_dir, "raw_trajectories.npy")
    rl_path = rl_path or os.path.join(data_dir, "rl_trajectories.npy")

    if not os.path.isfile(expert_path):
        sys.exit(f"Expert trajectories not found: {expert_path}\nRun Phase 1 first.")
    if not os.path.isfile(rl_path):
        sys.exit(
            f"RL rollouts not found: {rl_path}\n"
            "Run Step 3 first:  python run_collect_rollouts.py"
        )

    expert_raw = np.load(expert_path, allow_pickle=True)
    rl_raw = np.load(rl_path, allow_pickle=True)
    if max_expert is not None:
        expert_raw = expert_raw[:max_expert]

    print(f"[data] A* expert paths : {len(expert_raw)}")
    print(f"[data] PPO RL paths    : {len(rl_raw)}")

    expert_proc = process_raw_paths(expert_raw, horizon)
    rl_proc = process_raw_paths(rl_raw, horizon)

    if include_all_rl:
        merged = np.concatenate([expert_proc, rl_proc], axis=0)
    else:
        merged = expert_proc

    print(
        f"[data] Merged dataset  : {merged.shape}  "
        f"({len(expert_proc)} expert + {len(rl_proc)} RL)"
    )

    rng = np.random.default_rng(seed)
    n = len(merged)
    n_val = max(1, int(n * val_split))
    idx = rng.permutation(n)
    train_data = merged[idx[n_val:]]
    val_data = merged[idx[:n_val]]

    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, TRAIN_OUT)
    val_path = os.path.join(data_dir, VAL_OUT)
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    print(f"[data] Saved train → {train_path}  {train_data.shape}")
    print(f"[data] Saved val   → {val_path}  {val_data.shape}")

    stats = {
        "n_expert": len(expert_proc),
        "n_rl": len(rl_proc),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "horizon": horizon,
        "train_path": train_path,
        "val_path": val_path,
    }
    return train_data, val_data, stats


def _plot_loss(train_losses, val_losses, out_path: str):
    if not _PLT:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_losses, label="Train loss", linewidth=1.5)
    ax.plot(val_losses, label="Val loss", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Phase 2.5 — Diffusion Fine-tune (A* + RL rollouts)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Loss curve → {out_path}")


def finetune(
    *,
    data_dir: str | None = None,
    checkpoint_dir: str | None = None,
    results_dir: str | None = None,
    init_ckpt: str | None = None,
    out_ckpt_name: str = FINETUNED_CKPT,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 5e-5,
    horizon: int = 16,
    device: str | None = None,
    log_every: int = 10,
    seed: int = 42,
    val_split: float = 0.1,
    rebuild_dataset: bool = True,
    expert_path: str | None = None,
    rl_path: str | None = None,
) -> dict:
    data_dir = data_dir or os.path.join(_ROOT, "data")
    checkpoint_dir = checkpoint_dir or os.path.join(_ROOT, "checkpoints")
    results_dir = results_dir or os.path.join(_ROOT, "results")
    init_ckpt = init_ckpt or os.path.join(checkpoint_dir, "diffusion_best.pt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isfile(init_ckpt):
        sys.exit(f"Init checkpoint not found: {init_ckpt}\nRun Phase 2 first.")

    if device is None:
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    print(f"[finetune] Device: {device}")
    print(f"[finetune] Init weights: {init_ckpt}")

    if rebuild_dataset:
        train_data, val_data, ds_stats = build_merged_dataset(
            data_dir=data_dir,
            expert_path=expert_path,
            rl_path=rl_path,
            horizon=horizon,
            val_split=val_split,
            seed=seed,
        )
    else:
        train_path = os.path.join(data_dir, TRAIN_OUT)
        val_path = os.path.join(data_dir, VAL_OUT)
        if not os.path.isfile(train_path):
            sys.exit(f"Merged train set not found: {train_path}")
        train_data = np.load(train_path).astype(np.float32)
        val_data = np.load(val_path).astype(np.float32)
        ds_stats = {"n_train": len(train_data), "n_val": len(val_data)}

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_data)),
        batch_size=batch_size,
        shuffle=False,
    )

    ckpt = torch.load(init_ckpt, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    T = cfg["T"]
    schedule = cfg["schedule"]
    channels = tuple(cfg["channels"])
    time_embed_dim = cfg["time_embed_dim"]
    state_dim = cfg["state_dim"]

    noise_sched = NoiseSchedule(T=T, schedule=schedule, device=device)
    unet = TemporalUNet(
        state_dim=state_dim,
        horizon=cfg["horizon"],
        channels=channels,
        time_embed_dim=time_embed_dim,
        context_dim=cfg.get("context_dim", CONTEXT_DIM),
    )
    unet.load_state_dict(ckpt["model_state"])
    diffusion = GaussianDiffusion(unet, noise_sched, device=device)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"[model] Loaded TemporalUNet — {n_params:,} params")
    print(f"[finetune] {epochs} epochs, lr={lr}, batch={batch_size}\n")

    optimiser = torch.optim.AdamW(unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    out_path = os.path.join(checkpoint_dir, out_ckpt_name)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
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
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train={epoch_train:.5f}  val={epoch_val:.5f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if epoch_val < best_val:
            best_val = epoch_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": unet.state_dict(),
                    "val_loss": best_val,
                    "init_ckpt": os.path.basename(init_ckpt),
                    "config": {
                        **cfg,
                        "finetuned": True,
                        "data_source": "astar+rl_rollouts",
                    },
                },
                out_path,
            )

    elapsed = time.time() - t0
    print(f"\n[finetune] Done in {elapsed:.0f}s. Best val loss: {best_val:.5f}")
    print(f"[finetune] Saved → {out_path}")

    return {
        **ds_stats,
        "best_val_loss": best_val,
        "checkpoint": out_path,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2.5 — Fine-tune diffusion on A* + RL data")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--skip_dataset", action="store_true",
                        help="Use existing trajectories_rl_finetune_*.npy")
    args = parser.parse_args()

    finetune(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        horizon=args.horizon,
        log_every=args.log_every,
        device=args.device,
        init_ckpt=args.init_ckpt,
        rebuild_dataset=not args.skip_dataset,
    )
