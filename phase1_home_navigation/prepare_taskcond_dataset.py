"""
prepare_taskcond_dataset.py — Phase 3 Diffusion dataset with 6-dim task context

Merges static A* + dynamic-LB routes; saves trajectories + laundry context vectors.

Outputs:
    data/trajectories_taskcond_train.npy   (N, H, 2)
    data/trajectories_taskcond_val.npy
    data/context_taskcond_train.npy        (N, 6)
    data/context_taskcond_val.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from phase1_home_navigation.prepare_trajectory_dataset import (
    GRID_SIZE,
    normalize,
    resample_path,
)
from phase2_diffusion.trajectory_utils import (
    goal_normalized,
    laundry_cell_from_meta,
    normalize_grid,
)


def _load_pairs(
    raw_path: str,
    meta_path: str | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list]:
    raw = np.load(raw_path, allow_pickle=True)
    meta = None
    if meta_path and os.path.isfile(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
    paths, contexts, metas = [], [], []
    goal_n = goal_normalized()
    for i, traj in enumerate(raw):
        path = np.array(traj, dtype=np.float32)
        m = meta[i] if meta is not None and i < len(meta) else None
        laundry = laundry_cell_from_meta(m, path)
        laundry_n = normalize_grid(laundry)
        start_n = normalize(path[0], GRID_SIZE)
        ctx = np.concatenate([start_n, goal_n, laundry_n], dtype=np.float32)
        paths.append(path)
        contexts.append(ctx)
        metas.append(m)
    return paths, contexts, metas


def prepare_taskcond_dataset(
    *,
    data_dir: str | None = None,
    horizon: int = 16,
    val_split: float = 0.1,
    seed: int = 42,
    include_dynamic: bool = True,
) -> dict:
    data_dir = data_dir or os.path.join(_ROOT, "data")
    static_raw = os.path.join(data_dir, "raw_trajectories.npy")
    static_meta = os.path.join(data_dir, "raw_trajectories_meta.npy")
    dynamic_raw = os.path.join(data_dir, "raw_trajectories_dynamic_pet.npy")
    dynamic_meta = os.path.join(data_dir, "raw_trajectories_dynamic_pet_meta.npy")

    if not os.path.isfile(static_raw):
        sys.exit(f"Missing {static_raw} — run Phase 1 first.")

    all_paths, all_ctx, sources = [], [], []

    p1, c1, _ = _load_pairs(static_raw, static_meta)
    all_paths.extend(p1)
    all_ctx.extend(c1)
    sources.extend(["static"] * len(p1))

    if include_dynamic and os.path.isfile(dynamic_raw):
        p2, c2, _ = _load_pairs(dynamic_raw, dynamic_meta)
        all_paths.extend(p2)
        all_ctx.extend(c2)
        sources.extend(["dynamic_pet"] * len(p2))
        print(f"[data] Merged static={len(p1)} + dynamic_pet={len(p2)}")
    else:
        print(f"[data] Static only: {len(p1)} paths")

    processed_traj, processed_ctx = [], []
    skipped = 0
    for path, ctx in zip(all_paths, all_ctx):
        if path.shape[0] < 2:
            skipped += 1
            continue
        path_norm = normalize(path, GRID_SIZE)
        processed_traj.append(resample_path(path_norm, horizon))
        processed_ctx.append(ctx)

    traj_arr = np.array(processed_traj, dtype=np.float32)
    ctx_arr = np.array(processed_ctx, dtype=np.float32)
    print(f"[data] Processed {traj_arr.shape}, context {ctx_arr.shape}, skipped={skipped}")

    rng = np.random.default_rng(seed)
    n = len(traj_arr)
    n_val = max(1, int(n * val_split))
    idx = rng.permutation(n)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    train_traj, val_traj = traj_arr[train_idx], traj_arr[val_idx]
    train_ctx, val_ctx = ctx_arr[train_idx], ctx_arr[val_idx]

    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "trajectories_taskcond_train.npy"), train_traj)
    np.save(os.path.join(data_dir, "trajectories_taskcond_val.npy"), val_traj)
    np.save(os.path.join(data_dir, "context_taskcond_train.npy"), train_ctx)
    np.save(os.path.join(data_dir, "context_taskcond_val.npy"), val_ctx)

    stats = {
        "n_train": len(train_traj),
        "n_val": len(val_traj),
        "horizon": horizon,
        "context_dim": 6,
    }
    print(f"[data] train={train_traj.shape}  val={val_traj.shape}")
    return stats


def main():
    p = argparse.ArgumentParser(description="Prepare task-conditioned Diffusion dataset")
    p.add_argument("--data_dir", type=str, default=os.path.join(_ROOT, "data"))
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--no_dynamic", action="store_true")
    args = p.parse_args()
    prepare_taskcond_dataset(
        data_dir=args.data_dir,
        horizon=args.horizon,
        val_split=args.val_split,
        include_dynamic=not args.no_dynamic,
    )


if __name__ == "__main__":
    main()
