"""
run_phase2_taskcond.py — Phase 2 extension: task-conditioned diffusion

Pipeline:
    1. Collect dynamic-LB A* routes (optional)
    2. Prepare 6-dim task-conditioned dataset
    3. Train diffusion_taskcond.pt (does not overwrite diffusion_best.pt)
    4. Multipath figures + open-loop report

Usage:
    python run_phase2_taskcond.py
    python run_phase2_taskcond.py --skip_collect --epochs 80
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def main():
    p = argparse.ArgumentParser(description="Phase 2 — task-conditioned diffusion")
    p.add_argument("--n_dynamic", type=int, default=200)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_eval_ep", type=int, default=100)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_prepare", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_vis", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    if args.device is None:
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("=" * 60)
    print("   Phase 2 — Task-Conditioned Diffusion Planner")
    print("=" * 60)
    print(f"   Device: {args.device}")
    print("   Keeps diffusion_best.pt + ppo_best.zip unchanged")
    print("=" * 60)

    log_path = os.path.join(_ROOT, "results", "phase2_taskcond_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    with open(log_path, "w") as f:
        f.write("Phase 2 taskcond run\n")

    if not args.skip_collect:
        log("\n[Step 1] Collecting dynamic-PET A* routes …")
        import numpy as np
        from phase1_home_navigation.collect_dynamic_pet_routes import collect_dynamic_pet_routes
        trajs, meta = collect_dynamic_pet_routes(args.n_dynamic)
        data_dir = os.path.join(_ROOT, "data")
        os.makedirs(data_dir, exist_ok=True)
        np.save(
            os.path.join(data_dir, "raw_trajectories_dynamic_pet.npy"),
            np.array(trajs, dtype=object),
            allow_pickle=True,
        )
        np.save(
            os.path.join(data_dir, "raw_trajectories_dynamic_pet_meta.npy"),
            np.array(meta, dtype=object),
            allow_pickle=True,
        )
        log(f"  Saved {len(trajs)} dynamic-PET routes")
    else:
        log("\n[Step 1] Skipped collect (--skip_collect)")

    if not args.skip_prepare:
        log("\n[Step 2] Preparing task-conditioned dataset …")
        from phase1_home_navigation.prepare_taskcond_dataset import prepare_taskcond_dataset
        stats = prepare_taskcond_dataset()
        log(f"  train={stats['n_train']}  val={stats['n_val']}  context_dim={stats['context_dim']}")
    else:
        log("\n[Step 2] Skipped prepare (--skip_prepare)")

    ckpt_path = os.path.join(_ROOT, "checkpoints", "diffusion_taskcond.pt")
    if not args.skip_train:
        log(f"\n[Step 3] Training task-conditioned diffusion ({args.epochs} epochs) …")
        from phase2_diffusion.train_diffusion_taskcond import train_taskcond
        init = os.path.join(_ROOT, "checkpoints", "diffusion_best.pt")
        ckpt_path = train_taskcond(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            init_ckpt=init if os.path.isfile(init) else None,
        )
        log(f"  Checkpoint → {ckpt_path}")
    else:
        log("\n[Step 3] Skipped train (--skip_train)")

    if not os.path.isfile(ckpt_path):
        sys.exit(f"[ERROR] Missing {ckpt_path} — run training first.")

    if not args.skip_vis:
        log("\n[Step 4] Generating multipath figures …")
        from phase2_diffusion.visualisation.plot_diffusion_multipath import run as run_multipath
        run_multipath(
            start_pos=(0, 3),
            n_samples=8,
            device=args.device,
            ckpt_name="diffusion_taskcond.pt",
            out_prefix="diffusion_taskcond",
        )
        log("  Figures → results/diffusion_taskcond_multipath_*.png")
    else:
        log("\n[Step 4] Skipped visualisation (--skip_vis)")

    if not args.skip_eval:
        log(f"\n[Step 5] Open-loop eval ({args.n_eval_ep} episodes) …")
        from phase2_diffusion.eval_diffusion_openloop import eval_openloop_ckpt, save_report
        rows = []
        for name in ("diffusion_best.pt", "diffusion_taskcond.pt"):
            path = os.path.join(_ROOT, "checkpoints", name)
            if not os.path.isfile(path):
                continue
            row = eval_openloop_ckpt(path, args.n_eval_ep, args.device)
            rows.append(row)
            log(
                f"  {name}: success={row['success_pct']:.1f}%  "
                f"hazard={row['mean_hazard_frac']:.2f}  "
                f"laundry_cov={row['mean_laundry_coverage']:.1f}%"
            )
        if rows:
            save_report(
                rows,
                os.path.join(_ROOT, "results", "diffusion_openloop_report.csv"),
                out_png=None,
            )
    else:
        log("\n[Step 5] Skipped eval (--skip_eval)")

    print("\n" + "=" * 60)
    print("   Phase 2 (taskcond) complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
