"""
evaluate.py  —  Phase 3 / Step 5

Fair comparison on the home grid (dynamic PET, laundry task):

    1. A* open-loop
    2. A* + replan
    3. Diffusion open-loop          (Phase 2 checkpoint)
    4. Diffusion + PPO              (Phase 2 + PPO)
    5. Diffusion-RL open-loop       (Phase 2.5 fine-tuned)
    6. Diffusion-RL + PPO           (Phase 2.5 + PPO)

Outputs:
    results/eval_success_rate.png
    results/eval_method_comparison.png   — same start→goal path comparison (3 methods)
    results/eval_ablation.png
    results/eval_summary.csv
    results/eval_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env import EscapeGameGridEnv, SPECIAL_TILES
from env.map_config import MIN_LAUNDRY_REQUIRED, SPECIAL_TILES as MAP_TILES
from phase1_home_navigation.home_route_planner import build_blocked_set, plan_home_route
from phase2_diffusion.checkpoint_names import (
    DIFFUSION_BEST,
    DIFFUSION_RL_FINETUNED,
    resolve_checkpoint,
)
from phase3_ppo.env_wrapper import DiffusionWaypointEnv
from phase3_ppo.success_metrics import is_task_success

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT = True
except ImportError:
    _PLT = False

GRID_SIZE = 12
DEFAULT_GOAL = tuple(MAP_TILES["G"][0])
DEFAULT_LAUNDRY = [tuple(c) for c in MAP_TILES["LB"]]
MAX_EVAL_STEPS = 600
DEFAULT_REPLAN_EVERY = None  # None = event-driven only (laundry + stuck)
STUCK_REPLAN_THRESHOLD = 6

# All methods (used internally for ablation CSV)
METHOD_ORDER = [
    "A* open-loop",
    "A* + replan",
    "Diffusion open-loop",
    "Diffusion + PPO",
    "Diffusion-RL open-loop",
    "Diffusion-RL + PPO",
]

# Main results plot: open-loop ablations excluded — diffusion is a planner, not a standalone policy
MAIN_METHOD_ORDER = [
    "A* open-loop",
    "A* + replan",
    "Diffusion + PPO",
    "Diffusion-RL + PPO",
]

DIFFUSION_METHOD_ORDER = [
    "Diffusion + PPO",
    "Diffusion-RL + PPO",
]

DEFAULT_PLOT_TITLE = "Task Success Rate — Diffusion Trajectory Planner + PPO (dynamic obstacle)"

METHOD_COLORS = {
    "A* open-loop": "#4C72B0",
    "A* + replan": "#55A868",
    "Diffusion open-loop": "#C44E52",
    "Diffusion + PPO": "#DD8452",
    "Diffusion-RL open-loop": "#8172B3",
    "Diffusion-RL + PPO": "#CCB974",
}


def _action_from_delta(dr: int, dc: int) -> int:
    """Map (Δrow, Δcol) to env action: 0=Up, 1=Down, 2=Right, 3=Left."""
    if dr == -1 and dc == 0:
        return 0
    if dr == 1 and dc == 0:
        return 1
    if dr == 0 and dc == 1:
        return 2
    if dr == 0 and dc == -1:
        return 3
    raise ValueError(f"Invalid grid step ({dr}, {dc})")


def _make_base_env() -> EscapeGameGridEnv:
    return EscapeGameGridEnv(
        grid_size=GRID_SIZE,
        special_tiles=SPECIAL_TILES,
        random_initialization=True,
        headless=True,
        dynamic_pet=True,
    )


def _step_toward_cell(env, obs, target: tuple[int, int]):
    """Take one Manhattan grid step toward target. Returns (obs, reward, done, ok)."""
    r0, c0 = int(obs[0]), int(obs[1])
    r1, c1 = int(target[0]), int(target[1])
    dr, dc = r1 - r0, c1 - c0

    if dr == 0 and dc == 0:
        return obs, 0.0, False, True

    # One axis per step (diffusion waypoints are not axis-aligned)
    if abs(dr) >= abs(dc) and dr != 0:
        action = 0 if dr < 0 else 1
    elif dc != 0:
        action = 2 if dc > 0 else 3
    else:
        action = 0 if dr < 0 else 1

    obs, reward, done, _ = env.step(action)
    return obs, float(reward), bool(done), True


def _blocked_dynamic(env: EscapeGameGridEnv, static_blocked: set) -> set:
    """Static blocked tiles plus the current dynamic PET cell."""
    blocked = set(static_blocked)
    if env.pet_pos is not None:
        blocked.add(tuple(env.pet_pos))
    return blocked


def _sync_wp_idx(waypoints: list, pos: tuple[int, int]) -> int:
    """Pick the next waypoint index on path closest to the robot's current cell."""
    r, c = int(pos[0]), int(pos[1])
    for i, (wr, wc) in enumerate(waypoints):
        if (int(wr), int(wc)) == (r, c) and i + 1 < len(waypoints):
            return i + 1
    best_idx = 1
    best_d = 1e9
    for i in range(1, len(waypoints)):
        wr, wc = int(waypoints[i][0]), int(waypoints[i][1])
        d = abs(wr - r) + abs(wc - c)
        if d < best_d:
            best_d = d
            best_idx = i
    return min(best_idx, len(waypoints) - 1)


def _run_astar_episode(
    env: EscapeGameGridEnv,
    blocked: set,
    *,
    enable_replan: bool = False,
    periodic_replan: int | None = None,
    min_laundry: int = MIN_LAUNDRY_REQUIRED,
    skip_reset: bool = False,
    laundry_cells: list | None = None,
) -> dict:
    """Execute one episode with A* grid paths (open-loop or PET-aware replan)."""
    laundry_cells = laundry_cells or DEFAULT_LAUNDRY
    if skip_reset:
        obs = np.array(env.state + [env._coin_mask()], dtype=np.int32)
    else:
        obs, _ = env.reset()
    start = (int(obs[0]), int(obs[1]))
    plan = plan_home_route(
        start=start,
        goal=DEFAULT_GOAL,
        laundry_cells=laundry_cells,
        blocked=blocked,
        min_laundry=min_laundry,
        grid_size=GRID_SIZE,
        optimize_order=True,
    )

    total_r, steps = 0.0, 0
    path = [[obs[0], obs[1]]]
    done = False

    if plan is None:
        return {
            "reward": -50.0,
            "length": 1,
            "success": 0,
            "path": np.array(path),
            "waypoints_done": 0,
        }

    def _replan_from_current() -> bool:
        nonlocal waypoints, wp_idx, steps_since_replan, stuck_steps, laundry_done
        pos = (int(obs[0]), int(obs[1]))
        plan_blocked = _blocked_dynamic(env, blocked) if enable_replan else blocked
        new_plan = plan_home_route(
            start=pos,
            goal=DEFAULT_GOAL,
            laundry_cells=laundry_cells,
            blocked=plan_blocked,
            min_laundry=min_laundry,
            grid_size=GRID_SIZE,
            optimize_order=True,
        )
        if new_plan is None:
            return False
        waypoints = new_plan["positions"]
        wp_idx = _sync_wp_idx(waypoints, pos)
        steps_since_replan = 0
        stuck_steps = 0
        laundry_done = sum(env.coins) >= min_laundry
        return True

    waypoints = plan["positions"]
    wp_idx = 1
    steps_since_replan = 0
    laundry_done = sum(env.coins) >= min_laundry
    stuck_steps = 0
    prev_pos = (int(obs[0]), int(obs[1]))

    while not done and steps < MAX_EVAL_STEPS:
        if wp_idx >= len(waypoints):
            if not enable_replan or not _replan_from_current():
                break

        if enable_replan:
            curr_laundry_done = sum(env.coins) >= min_laundry
            need_replan = (
                (curr_laundry_done and not laundry_done)
                or stuck_steps >= STUCK_REPLAN_THRESHOLD
            )
            if periodic_replan is not None and steps_since_replan >= periodic_replan:
                need_replan = True
            if need_replan:
                _replan_from_current()

        target = waypoints[wp_idx]
        obs, r, step_done, ok = _step_toward_cell(env, obs, target)
        if not ok:
            break
        total_r += r
        steps += 1
        path.append([obs[0], obs[1]])
        done = step_done
        steps_since_replan += 1

        curr_pos = (int(obs[0]), int(obs[1]))
        if curr_pos == prev_pos:
            stuck_steps += 1
        else:
            stuck_steps = 0
        prev_pos = curr_pos

        if curr_pos == target:
            wp_idx += 1

    return {
        "reward": total_r,
        "length": steps,
        "success": 1 if is_task_success(env, total_reward=total_r) else 0,
        "path": np.array(path),
        "waypoints_done": 0,
    }


def _run_diffusion_open_loop_episode(
    planner: DiffusionWaypointEnv,
    *,
    skip_reset: bool = False,
) -> dict:
    """Sample Diffusion waypoints once at reset, replay without PPO or replan."""
    from phase3_ppo.env_wrapper import REACH_RADIUS

    if skip_reset:
        base = planner.base_env
        waypoints = planner._waypoints.copy()
        horizon = planner.horizon
    else:
        planner.reset()
        base = planner.base_env
        waypoints = planner._waypoints.copy()
        horizon = planner.horizon

    total_r, steps = 0.0, 0
    path = [[base.state[0], base.state[1]]]
    done = False
    wp_idx = 1
    waypoints_reached = 0
    stuck_at_same = 0
    prev_cell = (int(base.state[0]), int(base.state[1]))

    while not done and steps < MAX_EVAL_STEPS and wp_idx < horizon:
        target = (
            int(np.clip(round(waypoints[wp_idx][0]), 0, GRID_SIZE - 1)),
            int(np.clip(round(waypoints[wp_idx][1]), 0, GRID_SIZE - 1)),
        )
        obs_base = np.array(base.state, dtype=np.int32)
        obs_base, r, step_done, ok = _step_toward_cell(base, obs_base, target)
        if not ok:
            break
        total_r += r
        steps += 1
        path.append([obs_base[0], obs_base[1]])
        done = step_done

        curr_cell = (int(obs_base[0]), int(obs_base[1]))
        if curr_cell == prev_cell:
            stuck_at_same += 1
            if stuck_at_same >= 12:
                break
        else:
            stuck_at_same = 0
        prev_cell = curr_cell

        tr, tc = target
        dist_wp = abs(curr_cell[0] - tr) + abs(curr_cell[1] - tc)
        if dist_wp <= REACH_RADIUS or curr_cell == target:
            waypoints_reached += 1
            wp_idx += 1

    plan_cells = [
        (
            int(np.clip(round(waypoints[i][0]), 0, GRID_SIZE - 1)),
            int(np.clip(round(waypoints[i][1]), 0, GRID_SIZE - 1)),
        )
        for i in range(horizon)
    ]

    return {
        "reward": total_r,
        "length": steps,
        "success": 1 if is_task_success(base, total_reward=total_r) else 0,
        "path": np.array(path),
        "waypoints_done": waypoints_reached,
        "waypoints": np.array(plan_cells, dtype=np.int32),
        "horizon": horizon,
    }


def _aggregate_episodes(episodes: list[dict], horizon: int | None = None) -> dict:
    return {
        "rewards": np.array([e["reward"] for e in episodes]),
        "lengths": np.array([e["length"] for e in episodes]),
        "successes": np.array([e["success"] for e in episodes]),
        "paths": [e["path"] for e in episodes],
        "waypoints_done": np.array([e.get("waypoints_done", 0) for e in episodes]),
        "horizon": horizon,
    }


def run_astar_open_loop(
    n_episodes: int,
    min_laundry: int = MIN_LAUNDRY_REQUIRED,
) -> dict:
    blocked = build_blocked_set(MAP_TILES, GRID_SIZE)
    env = _make_base_env()
    episodes = [
        _run_astar_episode(env, blocked, enable_replan=False, min_laundry=min_laundry)
        for _ in range(n_episodes)
    ]
    env.close()
    return _aggregate_episodes(episodes)


def run_astar_replan(
    n_episodes: int,
    periodic_replan: int | None = DEFAULT_REPLAN_EVERY,
    min_laundry: int = MIN_LAUNDRY_REQUIRED,
) -> dict:
    blocked = build_blocked_set(MAP_TILES, GRID_SIZE)
    env = _make_base_env()
    episodes = [
        _run_astar_episode(
            env,
            blocked,
            enable_replan=True,
            periodic_replan=periodic_replan,
            min_laundry=min_laundry,
        )
        for _ in range(n_episodes)
    ]
    env.close()
    return _aggregate_episodes(episodes)


def run_diffusion_open_loop(
    n_episodes: int,
    device: str = "cpu",
    diffusion_ckpt: str | None = None,
) -> dict | None:
    diffusion_ckpt = diffusion_ckpt or resolve_checkpoint(
        os.path.join(_ROOT, "checkpoints"), DIFFUSION_BEST
    )
    if not os.path.exists(diffusion_ckpt):
        print(f"[eval] Diffusion checkpoint not found at {diffusion_ckpt} — skipping")
        return None

    planner = DiffusionWaypointEnv(ckpt_path=diffusion_ckpt, device=device, random_start=True)
    horizon = planner.horizon
    episodes = [_run_diffusion_open_loop_episode(planner) for _ in range(n_episodes)]
    planner.close()
    return _aggregate_episodes(episodes, horizon=horizon)


def run_diffusion_ppo(
    n_episodes: int,
    ppo_ckpt: str,
    device: str = "cpu",
    diffusion_ckpt: str | None = None,
    seed: int = 42,
) -> dict | None:
    from stable_baselines3 import PPO as SB3PPO

    if not os.path.exists(ppo_ckpt):
        print(f"[eval] PPO checkpoint not found at {ppo_ckpt} — skipping")
        return None

    diffusion_ckpt = diffusion_ckpt or resolve_checkpoint(
        os.path.join(_ROOT, "checkpoints"), DIFFUSION_BEST
    )
    model = SB3PPO.load(ppo_ckpt, device=device)
    env = DiffusionWaypointEnv(
        ckpt_path=diffusion_ckpt, device=device, random_start=True,
    )

    episodes = []
    np.random.seed(seed)

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_r, steps = 0.0, 0
        path = [env.base_env.state[:]]
        done = False

        while not done and steps < MAX_EVAL_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_r += r
            steps += 1
            path.append(env.base_env.state[:])

        episodes.append({
            "reward": total_r,
            "length": steps,
            "success": 1 if is_task_success(env.base_env, total_reward=total_r) else 0,
            "path": np.array(path),
            "waypoints_done": info.get("waypoints_done", 0),
        })

    horizon = env.horizon
    env.close()
    return _aggregate_episodes(episodes, horizon=horizon)


def _method_order(results: dict, *, diffusion_only: bool) -> list[str]:
    order = DIFFUSION_METHOD_ORDER if diffusion_only else MAIN_METHOD_ORDER
    return [m for m in order if m in results and results[m] is not None]


def plot_success_bar(
    results: dict,
    out_path: str,
    title: str | None = None,
    *,
    diffusion_only: bool = False,
):
    if not _PLT:
        return

    names = _method_order(results, diffusion_only=diffusion_only)
    rates = [np.mean(results[m]["successes"]) * 100 for m in names]
    colors = [METHOD_COLORS[m] for m in names]

    width = max(9, len(names) * 1.35)
    fig, ax = plt.subplots(figsize=(width, 5))
    bars = ax.bar(names, rates, color=colors, edgecolor="black", width=0.55)
    for bar, r in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{r:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylim(0, 115)
    ax.set_ylabel("Strict Task Success (%)", fontsize=12)
    ax.set_title(
        title or DEFAULT_PLOT_TITLE,
        fontsize=13, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {out_path}")


def plot_ablation(results: dict, out_path: str):
    """Grouped bar: Diffusion vs Diffusion-RL for open-loop and +PPO."""
    if not _PLT:
        return

    pairs = [
        ("Open-loop", "Diffusion open-loop", "Diffusion-RL open-loop"),
        ("+ PPO", "Diffusion + PPO", "Diffusion-RL + PPO"),
    ]
    labels, base_rates, rl_rates = [], [], []
    for label, base_key, rl_key in pairs:
        if results.get(base_key) is None or results.get(rl_key) is None:
            continue
        labels.append(label)
        base_rates.append(np.mean(results[base_key]["successes"]) * 100)
        rl_rates.append(np.mean(results[rl_key]["successes"]) * 100)

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, base_rates, w, label="Diffusion (Phase 2)", color="#C44E52", edgecolor="black")
    ax.bar(x + w / 2, rl_rates, w, label="Diffusion-RL (Phase 2.5)", color="#8172B3", edgecolor="black")
    for i, (b, r) in enumerate(zip(base_rates, rl_rates)):
        ax.text(i - w / 2, b + 1, f"{b:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w / 2, r + 1, f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")
        delta = r - b
        ax.text(i, max(b, r) + 8, f"Δ {delta:+.1f}%", ha="center", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Strict Task Success (%)")
    ax.set_title("Ablation: RL-Informed Diffusion Fine-Tuning (Phase 2.5)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {out_path}")


def save_ablation_csv(results: dict, out_path: str):
    rows = [
        ("Diffusion open-loop", "Diffusion open-loop", "Diffusion-RL open-loop"),
        ("Diffusion + PPO", "Diffusion + PPO", "Diffusion-RL + PPO"),
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "setting", "diffusion_success_pct", "diffusion_rl_success_pct", "delta_pct",
            "diffusion_steps", "diffusion_rl_steps",
        ])
        for setting, base_key, rl_key in rows:
            base, rl = results.get(base_key), results.get(rl_key)
            if base is None or rl is None:
                continue
            b_sr = np.mean(base["successes"]) * 100
            r_sr = np.mean(rl["successes"]) * 100
            writer.writerow([
                setting,
                f"{b_sr:.1f}",
                f"{r_sr:.1f}",
                f"{r_sr - b_sr:+.1f}",
                f"{np.mean(base['lengths']):.1f}",
                f"{np.mean(rl['lengths']):.1f}",
            ])
    print(f"[eval] Ablation CSV → {out_path}")


def plot_reward_dist(
    results: dict,
    out_path: str,
    *,
    diffusion_only: bool = False,
):
    if not _PLT:
        return

    valid = {k: v for k, v in results.items() if v is not None}
    names = _method_order(valid, diffusion_only=diffusion_only)
    width = max(9, len(names) * 1.35)
    fig, ax = plt.subplots(figsize=(width, 5))
    data = [valid[m]["rewards"] for m in names]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, notch=False)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(METHOD_COLORS[name])
        patch.set_alpha(0.75)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(DEFAULT_PLOT_TITLE, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {out_path}")


def plot_path_comparison(
    results: dict,
    out_path: str,
    n: int = 4,
    *,
    diffusion_only: bool = False,
):
    if not _PLT:
        return

    valid = {k: v for k, v in results.items() if v is not None}
    names = _method_order(valid, diffusion_only=diffusion_only)
    if len(names) < 2:
        return

    fig, axes = plt.subplots(len(names), n, figsize=(n * 2.8, len(names) * 2.2))
    if len(names) == 1:
        axes = np.array([axes])

    for row_idx, name in enumerate(names):
        res = valid[name]
        color = METHOD_COLORS[name]
        for i in range(n):
            ax = axes[row_idx, i]
            path = res["paths"][i] if i < len(res["paths"]) else np.zeros((2, 2))
            ax.plot(path[:, 1], path[:, 0], "-o", color=color, markersize=2, linewidth=1.5)
            ax.scatter(path[0, 1], path[0, 0], color="lime", s=40, zorder=5)
            ax.scatter(path[-1, 1], path[-1, 0], color="red", s=40, zorder=5)
            ax.scatter(10, 10, marker="*", color="gold", s=100, zorder=6)
            ax.set_xlim(-0.5, GRID_SIZE - 0.5)
            ax.set_ylim(GRID_SIZE - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            mark = "✓" if res["successes"][i] else "✗"
            ax.set_title(f"Ep {i + 1} {mark}", fontsize=8)
            if i == 0:
                ax.set_ylabel(name, fontsize=9, fontweight="bold")

    fig.suptitle(DEFAULT_PLOT_TITLE, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] → {out_path}")


def save_summary_csv(
    results: dict,
    out_path: str,
    *,
    diffusion_only: bool = False,
):
    order = METHOD_ORDER  # always save all methods to CSV for reference
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "success_pct", "mean_reward", "mean_steps", "mean_waypoints",
        ])
        for name in order:
            res = results.get(name)
            if res is None:
                continue
            sr = np.mean(res["successes"]) * 100
            mr = np.mean(res["rewards"])
            ml = np.mean(res["lengths"])
            mw = np.mean(res["waypoints_done"])
            h = res.get("horizon")
            wp_str = f"{mw:.1f}/{h}" if h else f"{mw:.1f}"
            writer.writerow([name, f"{sr:.1f}", f"{mr:.2f}", f"{ml:.1f}", wp_str])
    print(f"[eval] Summary CSV → {out_path}")


def print_summary(name: str, res: dict | None):
    if res is None:
        print(f"  {name:20s}  NOT AVAILABLE")
        return
    sr = np.mean(res["successes"]) * 100
    mr = np.mean(res["rewards"])
    ml = np.mean(res["lengths"])
    print(f"  {name:20s}  success={sr:5.1f}%  mean_reward={mr:+7.1f}  mean_steps={ml:.0f}")
    if res.get("horizon"):
        mw = np.mean(res["waypoints_done"])
        print(f"  {'':20s}  mean_waypoints_followed={mw:.1f} / {res['horizon']}")


def save_plots(
    results: dict,
    results_dir: str,
    *,
    diffusion_only: bool = False,
) -> None:
    """Write eval figures and CSV from an in-memory results dict."""
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return
    kw = dict(diffusion_only=diffusion_only)
    plot_success_bar(
        valid, os.path.join(results_dir, "eval_success_rate.png"), **kw,
    )
    plot_reward_dist(
        valid, os.path.join(results_dir, "eval_reward_dist.png"), **kw,
    )
    save_summary_csv(
        valid, os.path.join(results_dir, "eval_summary.csv"), **kw,
    )


def evaluate(
    n_episodes: int = 100,
    device: str = "cpu",
    checkpoint_dir: str | None = None,
    results_dir: str | None = None,
    periodic_replan: int | None = DEFAULT_REPLAN_EVERY,
    ppo_ckpt: str | None = None,
    diffusion_ckpt: str | None = None,
    diffusion_rl_ckpt: str | None = None,
    include_rl_finetuned: bool = True,
    diffusion_only: bool = False,
) -> dict:
    checkpoint_dir = checkpoint_dir or os.path.join(_ROOT, "checkpoints")
    results_dir = results_dir or os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    ppo_ckpt = ppo_ckpt or os.path.join(checkpoint_dir, "ppo_best.zip")
    diffusion_ckpt = diffusion_ckpt or resolve_checkpoint(checkpoint_dir, DIFFUSION_BEST)
    diffusion_rl_ckpt = diffusion_rl_ckpt or resolve_checkpoint(
        checkpoint_dir, DIFFUSION_RL_FINETUNED
    )

    has_rl = include_rl_finetuned and os.path.isfile(diffusion_rl_ckpt)
    n_methods = (4 if has_rl else 2) if diffusion_only else (6 if has_rl else 4)
    label = "Diffusion-only" if diffusion_only else "Full"
    print(f"\n[eval] {label} evaluation — {n_episodes} episodes × {n_methods} methods")
    if not diffusion_only:
        if periodic_replan is not None:
            print(f"[eval] A* replan: periodic every {periodic_replan} steps + laundry/stuck")
        else:
            print("[eval] A* replan: after laundry pickup, when stuck, path end (PET-aware)")
    print()

    all_results: dict = {}
    step = 1

    if not diffusion_only:
        print(f"[eval] {step}/{n_methods} A* open-loop …"); step += 1
        all_results["A* open-loop"] = run_astar_open_loop(n_episodes)
        print(f"[eval] {step}/{n_methods} A* + replan …"); step += 1
        all_results["A* + replan"] = run_astar_replan(
            n_episodes, periodic_replan=periodic_replan,
        )

    print(f"[eval] {step}/{n_methods} Diffusion open-loop …"); step += 1
    all_results["Diffusion open-loop"] = run_diffusion_open_loop(
        n_episodes, device=device, diffusion_ckpt=diffusion_ckpt,
    )

    print(f"[eval] {step}/{n_methods} Diffusion + PPO …"); step += 1
    all_results["Diffusion + PPO"] = run_diffusion_ppo(
        n_episodes, ppo_ckpt, device=device, diffusion_ckpt=diffusion_ckpt,
    )

    if has_rl:
        print(f"[eval] {step}/{n_methods} Diffusion-RL open-loop …"); step += 1
        all_results["Diffusion-RL open-loop"] = run_diffusion_open_loop(
            n_episodes, device=device, diffusion_ckpt=diffusion_rl_ckpt,
        )
        print(f"[eval] {step}/{n_methods} Diffusion-RL + PPO …")
        all_results["Diffusion-RL + PPO"] = run_diffusion_ppo(
            n_episodes, ppo_ckpt, device=device, diffusion_ckpt=diffusion_rl_ckpt,
        )
    elif include_rl_finetuned:
        print(f"[eval] Skipping Diffusion-RL methods — not found: {diffusion_rl_ckpt}")

    print_order = DIFFUSION_METHOD_ORDER if diffusion_only else METHOD_ORDER
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS (strict task success)")
    print("=" * 60)
    for name in print_order:
        if name in all_results:
            print_summary(name, all_results.get(name))
    print("=" * 60 + "\n")

    save_plots(all_results, results_dir, diffusion_only=diffusion_only)

    from phase3_ppo.plot_method_comparison import (
        generate_method_comparison,
        load_eval_success_rates,
    )

    success_rates = load_eval_success_rates(results_dir)
    if not success_rates:
        for name, res in all_results.items():
            if res and "successes" in res:
                success_rates[name] = float(np.mean(res["successes"]) * 100)

    generate_method_comparison(
        results_dir,
        device=device,
        ppo_ckpt=ppo_ckpt,
        diffusion_ckpt=diffusion_ckpt,
        periodic_replan=periodic_replan,
        success_rates=success_rates,
    )

    print("[eval] All evaluation complete.")
    return all_results


# Backward-compatible aliases
run_astar_expert = run_astar_open_loop


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fair eval: A* vs Diffusion baselines + PPO")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--periodic_replan",
        type=int,
        default=-1,
        help="Optional fixed replan interval in steps (-1 = event-driven only)",
    )
    p.add_argument("--ppo_ckpt", type=str, default=None)
    p.add_argument("--diffusion_ckpt", type=str, default=None)
    p.add_argument("--diffusion_rl_ckpt", type=str, default=None)
    p.add_argument("--baseline_only", action="store_true",
                   help="Skip Diffusion-RL methods (2-method diffusion eval only)")
    p.add_argument("--include_astar", action="store_true",
                   help="Include A* open-loop and A* + replan in eval and plots")
    args = p.parse_args()
    periodic = None if args.periodic_replan < 0 else args.periodic_replan
    evaluate(
        n_episodes=args.n_episodes,
        device=args.device,
        periodic_replan=periodic,
        ppo_ckpt=args.ppo_ckpt,
        diffusion_ckpt=args.diffusion_ckpt,
        diffusion_rl_ckpt=args.diffusion_rl_ckpt,
        include_rl_finetuned=not args.baseline_only,
        diffusion_only=not args.include_astar,
    )
