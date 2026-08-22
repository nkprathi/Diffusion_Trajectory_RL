"""
eval_diffusion_openloop.py — Phase 3 Diffusion open-loop quality report
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from phase3_ppo.env_wrapper import DiffusionWaypointEnv, GRID_SIZE, STATIC_HAZARD_TILES
from phase3_ppo.evaluate import run_diffusion_open_loop

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except ImportError:
    _PLT = False


def _plan_metrics_from_env(env: DiffusionWaypointEnv) -> dict:
    wps = env._waypoints
    hazard = 0
    for wp in wps:
        r = int(np.clip(round(wp[0]), 0, GRID_SIZE - 1))
        c = int(np.clip(round(wp[1]), 0, GRID_SIZE - 1))
        if env.base_env.pet_pos is not None and (r, c) == env.base_env.pet_pos:
            hazard += 1
            continue
        if str(env.base_env.grid[r, c]) in STATIC_HAZARD_TILES:
            hazard += 1
    return {
        "hazard_frac": hazard / max(len(wps), 1),
        "laundry_coverage": float(env._plan_covers_laundry(wps)),
    }


def eval_openloop_ckpt(
    ckpt_path: str,
    n_episodes: int = 100,
    device: str = "cpu",
) -> dict:
    agg = run_diffusion_open_loop(n_episodes, device=device, diffusion_ckpt=ckpt_path)
    if agg is None:
        return {
            "ckpt": os.path.basename(ckpt_path),
            "success_pct": 0.0,
            "mean_steps": 0.0,
            "mean_hazard_frac": 1.0,
            "mean_laundry_coverage": 0.0,
            "n_episodes": n_episodes,
        }

    planner = DiffusionWaypointEnv(ckpt_path=ckpt_path, device=device, random_start=True)
    hazard_fracs, laundry_cov = [], []
    for _ in range(n_episodes):
        planner.reset()
        m = _plan_metrics_from_env(planner)
        hazard_fracs.append(m["hazard_frac"])
        laundry_cov.append(m["laundry_coverage"])
    planner.close()

    return {
        "ckpt": os.path.basename(ckpt_path),
        "success_pct": float(np.mean(agg["successes"]) * 100),
        "mean_steps": float(np.mean(agg["lengths"])),
        "mean_hazard_frac": float(np.mean(hazard_fracs)),
        "mean_laundry_coverage": float(np.mean(laundry_cov) * 100),
        "n_episodes": n_episodes,
    }


def save_report(rows: list[dict], out_csv: str, out_png: str | None = None):
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[report] CSV → {out_csv}")

    if _PLT and out_png:
        labels = [r["ckpt"] for r in rows]
        success = [r["success_pct"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#C44E52", "#8172B3", "#55A868"]
        ax.bar(labels, success, color=colors[: len(labels)], edgecolor="black")
        for i, v in enumerate(success):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Open-Loop Strict Success (%)")
        ax.set_title("Diffusion Open-Loop Quality (Phase 3)")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[report] Plot → {out_png}")


def main():
    p = argparse.ArgumentParser(description="Diffusion open-loop quality report")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--results_dir", type=str, default=os.path.join(_ROOT, "results"))
    args = p.parse_args()

    ckpt_dir = os.path.join(_ROOT, "checkpoints")
    rows = []
    for name in ("diffusion_best.pt", "diffusion_taskcond.pt"):
        path = os.path.join(ckpt_dir, name)
        if not os.path.isfile(path):
            print(f"[skip] {path}")
            continue
        print(f"\n[eval] Open-loop {name} …")
        row = eval_openloop_ckpt(path, args.n_episodes, args.device)
        rows.append(row)
        print(
            f"  success={row['success_pct']:.1f}%  steps={row['mean_steps']:.1f}  "
            f"hazard={row['mean_hazard_frac']:.2f}  laundry_cov={row['mean_laundry_coverage']:.1f}%"
        )

    if rows:
        os.makedirs(args.results_dir, exist_ok=True)
        save_report(
            rows,
            os.path.join(args.results_dir, "diffusion_openloop_report.csv"),
            out_png=None,
        )


if __name__ == "__main__":
    main()
