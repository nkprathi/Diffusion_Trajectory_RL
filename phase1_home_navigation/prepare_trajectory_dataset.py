"""
prepare_trajectory_dataset.py  —  Phase 1, Step 2

Convert raw home-navigation routes into a fixed-size Diffusion-ready dataset.

Steps:
  1. Load raw_trajectories.npy  (list of variable-length (row,col) paths)
  2. Normalize grid coords  (row, col) → [-1, 1]
  3. Resample each path to exactly H=16 waypoints via linear interpolation
  4. Split into train/val (90/10)
  5. Save as .npy files ready for the Diffusion training script

Usage:
    python phase1_home_navigation/prepare_trajectory_dataset.py
    python phase1_home_navigation/prepare_trajectory_dataset.py --horizon 32 --val_split 0.15

Output:
    data/trajectories_train.npy  shape (N_train, H, 2)
    data/trajectories_val.npy    shape (N_val, H, 2)
    data/data_stats.npy          normalization bounds
    results/data_distribution.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Grid size (must match env)
GRID_SIZE = 12


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize(coords: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Map grid integer coordinates [0, grid_size-1] → [-1, 1].

    Args:
        coords : (..., 2)  float array of (row, col) values
    Returns:
        (..., 2)  float array in [-1, 1]
    """
    return 2.0 * coords / (grid_size - 1) - 1.0


def denormalize(coords: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Inverse of normalize: [-1, 1] → [0, grid_size-1]."""
    return (coords + 1.0) / 2.0 * (grid_size - 1)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_path(path: np.ndarray, H: int) -> np.ndarray:
    """
    Uniformly resample a variable-length path to exactly H waypoints
    using linear interpolation along cumulative arc-length.

    Args:
        path : (N, 2)  raw waypoints (float)
        H    : target number of waypoints

    Returns:
        (H, 2)  resampled path
    """
    if len(path) < 2:
        return np.tile(path[0], (H, 1))

    diffs = np.diff(path, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total_len = cumlen[-1]

    if total_len < 1e-8:
        return np.tile(path[0], (H, 1))

    target = np.linspace(0.0, total_len, H)
    resampled = np.zeros((H, 2))
    for i, s in enumerate(target):
        idx = int(np.searchsorted(cumlen, s, side="right")) - 1
        idx = np.clip(idx, 0, len(path) - 2)
        frac = (s - cumlen[idx]) / max(float(dists[idx]), 1e-8)
        resampled[i] = path[idx] + frac * (path[idx + 1] - path[idx])
    return resampled


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare(
    raw_path: str = "../data/raw_trajectories.npy",
    out_dir:  str = "../data",
    horizon:  int = 16,
    val_split: float = 0.1,
) -> None:

    if not os.path.exists(raw_path):
        sys.exit(
            f"Raw trajectories not found at {raw_path}.\n"
            "Run  python run_phase1.py  or  python phase1_home_navigation/collect_expert_routes.py  first."
        )

    raw = np.load(raw_path, allow_pickle=True)
    print(f"Loaded {len(raw)} raw trajectories from {raw_path}")

    processed = []
    skipped = 0

    for traj in raw:
        # Convert list of (row, col) tuples → float array
        path = np.array(traj, dtype=np.float32)           # (T, 2)
        if path.shape[0] < 2:
            skipped += 1
            continue

        # Normalize to [-1, 1]
        path_norm = normalize(path, GRID_SIZE)

        # Resample to fixed horizon H
        path_resampled = resample_path(path_norm, horizon)  # (H, 2)
        processed.append(path_resampled)

    if skipped:
        print(f"Skipped {skipped} degenerate trajectories (< 2 steps).")

    dataset = np.array(processed, dtype=np.float32)        # (N, H, 2)
    print(f"Dataset shape: {dataset.shape}  (N trajectories × {horizon} waypoints × 2 coords)")
    print(f"Value range: [{dataset.min():.3f}, {dataset.max():.3f}]  (should be ~ [-1, 1])")

    # Train / val split
    n = len(dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    idx = np.random.permutation(n)
    train_data = dataset[idx[:n_train]]
    val_data   = dataset[idx[n_train:]]

    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "trajectories_train.npy")
    val_path   = os.path.join(out_dir, "trajectories_val.npy")
    np.save(train_path, train_data)
    np.save(val_path,   val_data)

    print(f"\nSaved training set  → {train_path}  {train_data.shape}")
    print(f"Saved validation set → {val_path}   {val_data.shape}")

    # Save normalization stats
    stats = {"grid_size": GRID_SIZE, "horizon": horizon}
    stats_path = os.path.join(out_dir, "data_stats.npy")
    np.save(stats_path, stats)
    print(f"Data stats saved     → {stats_path}")

    _plot_distribution(dataset, train_data, val_data, horizon)


# ---------------------------------------------------------------------------
# Diagnostics plot
# ---------------------------------------------------------------------------

def _plot_distribution(
    dataset: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    horizon: int,
) -> None:
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Phase 1 — Waypoint Density Heatmap", fontsize=13)

    all_rows = denormalize(dataset[:, :, 0], GRID_SIZE).flatten()
    all_cols = denormalize(dataset[:, :, 1], GRID_SIZE).flatten()
    heatmap, _, _ = np.histogram2d(all_rows, all_cols, bins=12, range=[[0, 12], [0, 12]])
    ax.imshow(heatmap, origin="upper", cmap="hot")
    ax.set_title("Expert Route Waypoint Density")
    ax.set_xlabel("Col")
    ax.set_ylabel("Row")

    plt.tight_layout()
    path = os.path.join(results_dir, "data_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Data distribution plot saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Prepare Diffusion trajectory dataset.")
    parser.add_argument("--raw_path",  default=os.path.join(_ROOT, "data", "raw_trajectories.npy"))
    parser.add_argument("--out_dir",   default=os.path.join(_ROOT, "data"))
    parser.add_argument("--horizon",   type=int,   default=16,
                        help="Number of waypoints per trajectory (default 16)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction held out for validation (default 0.1)")
    args = parser.parse_args()

    prepare(
        raw_path=args.raw_path,
        out_dir=args.out_dir,
        horizon=args.horizon,
        val_split=args.val_split,
    )
