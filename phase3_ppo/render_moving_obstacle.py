"""
render_moving_obstacle.py — MP4 video of dynamic PET near the robot.

Output (under results/moving_obstacle/):
  moving_obstacle.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from env.dynamic_obstacle import cells_within_radius
from env.grid_viz import draw_home_grid, row_col_center
from env.map_config import DYNAMIC_PET_RADIUS, GRID_SIZE, SPECIAL_TILES

OUT_DIR = os.path.join(_ROOT, "results", "moving_obstacle")
DEFAULT_VIDEO_START = (0, 3)
MASK_START_CELL = {tuple(SPECIAL_TILES["S"][0])}


def render_snapshot(
    ax,
    *,
    robot: tuple[int, int],
    pet_pos: tuple[int, int] | None,
    step: int,
    title: str,
) -> None:
    draw_home_grid(
        ax,
        title=title,
        show_legend=False,
        show_axis_labels=True,
        mask_cells=MASK_START_CELL,
    )

    zone = cells_within_radius(robot, DYNAMIC_PET_RADIUS, GRID_SIZE)
    for r, c in zone:
        ax.add_patch(
            mpatches.Rectangle(
                (c, r), 1, 1,
                facecolor="#E8DAEF", edgecolor="none", alpha=0.45, zorder=2,
            )
        )

    rx, ry = row_col_center(*robot)
    ax.plot(rx, ry, "o", color="#27AE60", markersize=14,
            markeredgecolor="white", markeredgewidth=1.2, zorder=7)

    if pet_pos is not None:
        px, py = row_col_center(*pet_pos)
        ax.plot(px, py, "s", color="#8E44AD", markersize=15,
                markeredgecolor="white", markeredgewidth=1.2, zorder=8)
        ax.text(px, py, "PET", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=9)

    ax.text(
        0.02, 0.98, f"step {step}",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )


def _fig_to_array(fig) -> np.ndarray:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def record_episode(
    *,
    seed: int = 42,
    max_steps: int = 40,
    device: str = "cpu",
    ppo_ckpt: str | None = None,
    diffusion_ckpt: str | None = None,
    fixed_start: tuple[int, int] = DEFAULT_VIDEO_START,
) -> list[dict]:
    from stable_baselines3 import PPO
    from phase3_ppo.env_wrapper import DiffusionWaypointEnv

    ppo_ckpt = ppo_ckpt or os.path.join(_ROOT, "checkpoints", "ppo_best.zip")
    if not os.path.exists(ppo_ckpt):
        raise FileNotFoundError(f"PPO checkpoint not found: {ppo_ckpt}")

    diffusion_ckpt = diffusion_ckpt or os.path.join(_ROOT, "checkpoints", "diffusion_rl_finetuned.pt")
    if not os.path.isfile(diffusion_ckpt):
        diffusion_ckpt = os.path.join(_ROOT, "checkpoints", "diffusion_best.pt")

    wrap = DiffusionWaypointEnv(
        ckpt_path=diffusion_ckpt, device=device, random_start=False,
    )
    model = PPO.load(ppo_ckpt, device=device)
    reset_opts = {"fixed_start": fixed_start, "pet_seed": seed}
    obs, _ = wrap.reset(seed=seed, options=reset_opts)

    snapshots = [{
        "step": 0,
        "robot": tuple(wrap.base_env.state),
        "pet_pos": wrap.base_env.pet_pos,
    }]

    done = False
    step = 0
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = wrap.step(int(action))
        done = terminated or truncated
        step += 1
        snapshots.append({
            "step": step,
            "robot": tuple(wrap.base_env.state),
            "pet_pos": wrap.base_env.pet_pos,
        })

    wrap.close()
    return snapshots


def _cleanup_stale_outputs(out_dir: str) -> None:
    """Remove legacy timestep PNGs and frame dumps."""
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if name.startswith("timestep_") and name.endswith(".png"):
            os.remove(path)
        elif name == "moving_obstacle_timesteps.png" and os.path.isfile(path):
            os.remove(path)
        elif name == "frames" and os.path.isdir(path):
            shutil.rmtree(path)


def render_moving_obstacle(
    out_dir: str = OUT_DIR,
    seed: int = 42,
    max_steps: int = 40,
    fps: int = 3,
    device: str = "cpu",
    ppo_ckpt: str | None = None,
    diffusion_ckpt: str | None = None,
    fixed_start: tuple[int, int] = DEFAULT_VIDEO_START,
) -> str | None:
    os.makedirs(out_dir, exist_ok=True)
    _cleanup_stale_outputs(out_dir)

    data = record_episode(
        seed=seed, max_steps=max_steps, device=device,
        ppo_ckpt=ppo_ckpt, diffusion_ckpt=diffusion_ckpt,
        fixed_start=fixed_start,
    )

    video_frames: list[np.ndarray] = []
    for item in data:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("white")
        render_snapshot(
            ax,
            robot=item["robot"],
            pet_pos=item["pet_pos"],
            step=item["step"],
            title=f"Dynamic pet obstacle  (radius r={DYNAMIC_PET_RADIUS})",
        )
        fig.tight_layout()
        video_frames.append(_fig_to_array(fig))
        plt.close(fig)

    video_path = os.path.join(out_dir, "moving_obstacle.mp4")
    try:
        import imageio.v2 as imageio

        imageio.mimsave(video_path, video_frames, fps=fps, codec="libx264")
        print(f"Saved → {video_path}")
        return video_path
    except Exception as exc:
        print(f"[warn] MP4 export failed: {exc}")
        return None


def main():
    p = argparse.ArgumentParser(description="Render moving pet obstacle MP4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ppo_ckpt", type=str, default=None)
    p.add_argument("--diffusion_ckpt", type=str, default=None)
    p.add_argument("--start", type=int, nargs=2, default=list(DEFAULT_VIDEO_START),
                   metavar=("ROW", "COL"))
    args = p.parse_args()
    render_moving_obstacle(
        out_dir=args.out_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        fps=args.fps,
        device=args.device,
        ppo_ckpt=args.ppo_ckpt,
        diffusion_ckpt=args.diffusion_ckpt,
        fixed_start=tuple(args.start),
    )


if __name__ == "__main__":
    main()
