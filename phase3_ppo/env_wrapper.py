"""
env_wrapper.py  —  Phase 3

DiffusionWaypointEnv: wraps EscapeGameGridEnv so that the PPO agent
navigates by following Diffusion-generated waypoints as subgoals.

Features:
  - Goal-conditioned diffusion sampling (start + goal context)
  - Plan resampling: reject trajectories with hazards on waypoints
  - Sequential waypoints: bonus only when advancing wp_idx in order
  - Stall penalty when waypoint index does not progress
  - Phase 2: event-driven replan (stuck, PET, path end, laundry)
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Set

import gymnasium as gym
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env import EscapeGameGridEnv, SPECIAL_TILES
from env.map_config import MIN_LAUNDRY_REQUIRED
from phase1_home_navigation.home_route_planner import build_blocked_set, plan_home_route
from phase2_diffusion.checkpoint_names import DIFFUSION_BEST, resolve_checkpoint
from phase2_diffusion.noise_schedule import NoiseSchedule
from phase2_diffusion.unet import TemporalUNet
from phase2_diffusion.diffusion import GaussianDiffusion
from phase2_diffusion.trajectory_utils import (
    make_context,
    make_task_context,
    goal_normalized,
    normalize_grid,
    nearest_laundry_cell,
)
from phase3_ppo.success_metrics import is_task_success

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE    = 12
DEFAULT_HORIZON = 16
# Phase 2.4: slightly larger capture radius + stronger progress shaping
REACH_RADIUS = 2.0
# Phase 1: reduce waypoint shaping, emphasize task completion (aligned with eval success)
WAYPOINT_BONUS = 25.0
WAYPOINT_BONUS_LAUNDRY = 8.0
WAYPOINT_BONUS_DELIVERY = 15.0
DISTANCE_SHAPING_COEF = 1.2
GOAL_SHAPING_COEF     = 0.0
GOAL_SHAPING_DELIVERY = 1.0
LAUNDRY_PICKUP_BONUS = 25.0
LAUNDRY_PLAN_RADIUS = 2.0
GOAL_COMPLETION_BONUS = 400.0
TIMEOUT_PENALTY = -75.0
MAX_STEPS    = 350
GOAL_POS     = (10, 10)
GOAL_RADIUS  = 1.5
OBS_DIM = 20

# Phase 2: closed-loop replanning (mirrors fair A* + replan eval triggers)
STUCK_REPLAN_THRESHOLD = 6
PET_REPLAN_NEAR_RADIUS = 2      # increased: replan when PET within 2 cells (was 1)
REPLAN_COOLDOWN_STEPS = 4
PERIODIC_REPLAN_EVERY = 10      # Way 2: replan every N steps even if not stuck

# Static tiles that invalidate a generated plan if a waypoint lies on them
STATIC_HAZARD_TILES: Set[str] = {"W", "F"}
# Dynamic PET is checked separately at execution time via hazard window
EXECUTION_HAZARD_TILES: Set[str] = {"W", "F", "PET"}

PLAN_RESAMPLE_ATTEMPTS = 15
MAX_HAZARD_FRAC = 0.15       # allow minor interpolation error on a few waypoints
GUIDANCE_SCALE = 1.0         # classifier-free guidance strength (1.0 = off)
STALL_STEP_LIMIT = 35        # steps without wp advance → penalty
STALL_PENALTY = 1.5
SEQUENTIAL_PROGRESS_BONUS = 2.0  # extra on top of waypoint_bonus per index


def _load_diffusion(ckpt_path: str, device: str) -> tuple[GaussianDiffusion, dict]:
    """Load trained diffusion planner from checkpoint; returns diffusion + config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]

    schedule = NoiseSchedule(T=cfg["T"], schedule=cfg["schedule"], device=device)
    context_dim = cfg.get("context_dim", 0)
    unet = TemporalUNet(
        state_dim=cfg["state_dim"],
        horizon=cfg["horizon"],
        channels=tuple(cfg["channels"]),
        time_embed_dim=cfg["time_embed_dim"],
        context_dim=context_dim,
    )
    unet.load_state_dict(ckpt["model_state"])
    unet.eval()
    return GaussianDiffusion(unet, schedule, device=device), cfg


def _denorm(coords: np.ndarray) -> np.ndarray:
    """[-1, 1] → [0, GRID_SIZE - 1]."""
    return (coords + 1.0) / 2.0 * (GRID_SIZE - 1)


def _resample_path_grid(path: np.ndarray, horizon: int) -> np.ndarray:
    """Uniform arc-length resample of grid (row,col) path to fixed horizon."""
    path = np.asarray(path, dtype=np.float32)
    if len(path) < 2:
        return np.tile(path[0], (horizon, 1)).astype(np.float32)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return np.tile(path[0], (horizon, 1)).astype(np.float32)
    targets = np.linspace(0.0, total, horizon)
    out = np.zeros((horizon, 2), dtype=np.float32)
    for i, t in enumerate(targets):
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = min(max(idx, 0), len(path) - 2)
        frac = (t - cum[idx]) / max(cum[idx + 1] - cum[idx], 1e-6)
        out[i] = path[idx] + frac * (path[idx + 1] - path[idx])
    return out


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class DiffusionWaypointEnv(gym.Env):
    """
    Gymnasium environment that combines:
      - EscapeGameGridEnv  (the 12×12 escape game)
      - GaussianDiffusion  (goal-conditioned Diffusion trajectory planner)

    Observation space (Box, 20-dim, float32):
        [0-1]   pose (row, col normalised)
        [2]     coin bitmask / 255
        [3-4]   current waypoint
        [5]     waypoint progress
        [6]     dist to goal (normalised)
        [7]     steps remaining
        [8-16]  3×3 local hazard window
        [17]    laundry requirement satisfied (0/1)
        [18]    dist to nearest uncollected laundry (normalised)
        [19]    task phase: 0=laundry, 1=delivery
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        ckpt_path: str = None,
        device: str = "cpu",
        random_start: bool = True,
        render_mode: Optional[str] = None,
        waypoint_bonus: float = WAYPOINT_BONUS,
        reach_radius: float = REACH_RADIUS,
        max_steps: int = MAX_STEPS,
        distance_shaping_coef: float = DISTANCE_SHAPING_COEF,
        plan_resample_attempts: int = PLAN_RESAMPLE_ATTEMPTS,
        max_hazard_frac: float = MAX_HAZARD_FRAC,
        stall_step_limit: int = STALL_STEP_LIMIT,
        stall_penalty: float = STALL_PENALTY,
        guidance_scale: float = GUIDANCE_SCALE,
    ):
        super().__init__()
        self.device         = device
        self.waypoint_bonus = waypoint_bonus
        self.distance_shaping_coef = distance_shaping_coef
        self.reach_radius   = reach_radius
        self.max_steps      = max_steps
        self.render_mode    = render_mode
        self.plan_resample_attempts = plan_resample_attempts
        self.max_hazard_frac = max_hazard_frac
        self.stall_step_limit = stall_step_limit
        self.stall_penalty = stall_penalty
        self.guidance_scale = guidance_scale

        if ckpt_path is None:
            ckpt_path = resolve_checkpoint(
                os.path.join(_ROOT, "checkpoints"), DIFFUSION_BEST
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {ckpt_path}\n"
                "Run Phase 2 first:  python run_phase2.py"
            )
        self.diffusion, self._diffusion_cfg = _load_diffusion(ckpt_path, device)
        self.horizon = self._diffusion_cfg["horizon"]
        self.context_dim = self._diffusion_cfg.get("context_dim", 0)
        self._goal_norm = goal_normalized()
        print(
            f"[env] Loaded diffusion planner from {ckpt_path} "
            f"(H={self.horizon}, goal_conditioned={self.context_dim > 0}, "
            f"plan_resample={self.plan_resample_attempts}, "
            f"guidance_scale={self.guidance_scale})"
        )

        headless = (render_mode != "human")
        self.base_env = EscapeGameGridEnv(
            grid_size=GRID_SIZE,
            special_tiles=SPECIAL_TILES,
            random_initialization=random_start,
            headless=headless,
            dynamic_pet=True,
        )

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self._max_dist = float(np.sqrt(2) * (GRID_SIZE - 1))  # max grid diagonal
        self.action_space = gym.spaces.Discrete(4)

        self._waypoints: np.ndarray = np.zeros((self.horizon, 2))
        self._wp_idx: int = 0
        self._steps: int = 0
        self._prev_dist: float = 0.0
        self._prev_dist_to_goal: float = 0.0
        self._stall_steps: int = 0
        self._last_wp_idx: int = 0
        self._prev_pos: tuple[int, int] = (0, 0)
        self._pos_stuck_steps: int = 0
        self._steps_since_replan: int = REPLAN_COOLDOWN_STEPS
        self._replan_count: int = 0
        self._blocked = build_blocked_set(SPECIAL_TILES, GRID_SIZE)
        self._laundry_cells = [tuple(c) for c in SPECIAL_TILES.get("LB", [])]

    def _laundry_satisfied(self) -> bool:
        return sum(self.base_env.coins) >= self.base_env.min_laundry_required

    def _nearest_laundry_cell(self, pos) -> Optional[tuple[int, int]]:
        if self._laundry_satisfied():
            return None
        best = None
        best_d = 1e9
        for idx, coord in enumerate(self.base_env.laundry_coords):
            if self.base_env.coins[idx]:
                continue
            d = float(np.linalg.norm(np.array(pos, dtype=np.float32) - np.array(coord, dtype=np.float32)))
            if d < best_d:
                best_d = d
                best = (int(coord[0]), int(coord[1]))
        return best

    def _plan_covers_laundry(self, waypoints: np.ndarray) -> bool:
        """True if some waypoint passes near an uncollected laundry pile."""
        if self._laundry_satisfied():
            return True
        for wp in waypoints:
            for idx, coord in enumerate(self.base_env.laundry_coords):
                if self.base_env.coins[idx]:
                    continue
                d = float(np.linalg.norm(wp - np.array(coord, dtype=np.float32)))
                if d <= LAUNDRY_PLAN_RADIUS:
                    return True
        return False

    def _blocked_with_pet(self) -> set:
        """Static blocked tiles plus the current dynamic PET cell."""
        blocked = set(self._blocked)
        if self.base_env.pet_pos is not None:
            blocked.add(tuple(self.base_env.pet_pos))
        return blocked

    def _sync_wp_idx(self, pos) -> int:
        """Pick the next waypoint index closest to the robot's current cell."""
        r, c = int(pos[0]), int(pos[1])
        for i, (wr, wc) in enumerate(self._waypoints):
            if (int(wr), int(wc)) == (r, c) and i + 1 < len(self._waypoints):
                return i + 1
        best_idx = 1
        best_d = 1e9
        for i in range(1, len(self._waypoints)):
            wr, wc = int(self._waypoints[i][0]), int(self._waypoints[i][1])
            d = abs(wr - r) + abs(wc - c)
            if d < best_d:
                best_d = d
                best_idx = i
        return min(best_idx, len(self._waypoints) - 1)

    def _on_or_near_pet(self, row: int, col: int) -> bool:
        pet = self.base_env.pet_pos
        if pet is None:
            return False
        pr, pc = pet
        if (row, col) == (pr, pc):
            return True
        return abs(row - pr) + abs(col - pc) <= PET_REPLAN_NEAR_RADIUS

    def _astar_plan_grid(
        self,
        start_pos: list,
        *,
        pet_aware: bool = False,
    ) -> Optional[np.ndarray]:
        """Expert multi-goal A* path resampled to diffusion horizon (grid coords)."""
        blocked = self._blocked_with_pet() if pet_aware else self._blocked
        plan = plan_home_route(
            start=(int(start_pos[0]), int(start_pos[1])),
            goal=GOAL_POS,
            laundry_cells=self._laundry_cells,
            blocked=blocked,
            min_laundry=MIN_LAUNDRY_REQUIRED,
            grid_size=GRID_SIZE,
            optimize_order=True,
        )
        if plan is None:
            return None
        path = np.array(plan["positions"], dtype=np.float32)
        wps = _resample_path_grid(path, self.horizon)
        wps[0] = np.array(start_pos, dtype=np.float32)
        wps[-1] = np.array(GOAL_POS, dtype=np.float32)
        return wps

    def _inject_laundry_anchor(self, waypoints: np.ndarray, start_pos: list) -> np.ndarray:
        """Force waypoint 1 toward nearest laundry when Diffusion plan misses it."""
        target = self._nearest_laundry_cell(start_pos)
        if target is None:
            return waypoints
        out = waypoints.copy()
        out[1] = np.array(target, dtype=np.float32)
        return out

    def _reset_waypoint_tracking(self, start_pos: list) -> None:
        self._wp_idx = self._sync_wp_idx(start_pos)
        self._stall_steps = 0
        self._prev_dist = float(np.linalg.norm(
            np.array(start_pos, dtype=np.float32) - self._waypoints[self._wp_idx]
        ))
        self._prev_dist_to_goal = float(np.linalg.norm(
            np.array(start_pos, dtype=np.float32) - np.array(GOAL_POS, dtype=np.float32)
        ))
        self._prev_laundry_dist = self._nearest_laundry_dist(start_pos)

    def _nearest_laundry_dist(self, pos) -> float:
        if sum(self.base_env.coins) >= self.base_env.min_laundry_required:
            return 0.0
        best = 1e9
        for idx, coord in enumerate(self.base_env.laundry_coords):
            if self.base_env.coins[idx]:
                continue
            d = float(np.linalg.norm(np.array(pos, dtype=np.float32) - np.array(coord, dtype=np.float32)))
            best = min(best, d)
        return best if best < 1e8 else 0.0

    def _tile_at(self, row: float, col: float) -> str:
        """Grid tile at floating (row, col), clipped to board."""
        r = int(np.clip(round(row), 0, GRID_SIZE - 1))
        c = int(np.clip(round(col), 0, GRID_SIZE - 1))
        if self.base_env.pet_pos is not None and (r, c) == self.base_env.pet_pos:
            return "PET"
        return str(self.base_env.grid[r, c])

    def _plan_hazard_fraction(self, waypoints: np.ndarray) -> float:
        hazards = 0
        for wp in waypoints:
            if self._tile_at(wp[0], wp[1]) in STATIC_HAZARD_TILES:
                hazards += 1
        return hazards / max(len(waypoints), 1)

    def _plan_is_valid(self, waypoints: np.ndarray) -> bool:
        """Reject plans with too many waypoints on bombs/walls/hazards."""
        return self._plan_hazard_fraction(waypoints) <= self.max_hazard_frac

    def _local_hazard_map(self, row: int, col: int) -> np.ndarray:
        """3×3 hazard window centered on (row, col). 1.0=hazard, 0.0=safe."""
        window = np.zeros(9, dtype=np.float32)
        pet_pos = self.base_env.pet_pos
        idx = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = row + dr, col + dc
                if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
                    window[idx] = 1.0  # out-of-bounds treated as wall
                elif pet_pos is not None and (r, c) == pet_pos:
                    window[idx] = 1.0
                elif str(self.base_env.grid[r, c]) in EXECUTION_HAZARD_TILES:
                    window[idx] = 1.0
                idx += 1
        return window

    def _make_obs(self, base_obs: np.ndarray) -> np.ndarray:
        row, col, coin_mask = base_obs
        wp = self._waypoints[self._wp_idx]
        denom = max(self.horizon - 1, 1)

        pos = np.array([row, col], dtype=np.float32)
        goal = np.array(GOAL_POS, dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(pos - goal)) / self._max_dist
        steps_remaining = max(0.0, (self.max_steps - self._steps) / self.max_steps)
        hazard_map = self._local_hazard_map(int(row), int(col))
        laundry_ok = self._laundry_satisfied()
        laundry_dist = self._nearest_laundry_dist([row, col])
        laundry_dist_norm = 0.0 if laundry_ok else min(laundry_dist / self._max_dist, 1.0)
        task_phase = 1.0 if laundry_ok else 0.0

        return np.array([
            row / (GRID_SIZE - 1),
            col / (GRID_SIZE - 1),
            coin_mask / 255.0,
            wp[0] / (GRID_SIZE - 1),
            wp[1] / (GRID_SIZE - 1),
            self._wp_idx / denom,
            dist_to_goal,
            steps_remaining,
            *hazard_map,
            float(laundry_ok),
            laundry_dist_norm,
            task_phase,
        ], dtype=np.float32)

    def _build_sample_context(
        self,
        start_pos: list,
        start_t: torch.Tensor,
        goal_t: torch.Tensor,
    ) -> torch.Tensor:
        """4-dim [start;goal] or 6-dim [start;goal;laundry] from checkpoint config."""
        if self.context_dim >= 6:
            laundry = nearest_laundry_cell(
                start_pos, collected_mask=list(self.base_env.coins),
            )
            laundry_t = torch.tensor(
                normalize_grid(laundry), device=self.device, dtype=torch.float32,
            ).unsqueeze(0)
            return make_task_context(start_t, goal_t, laundry_t)
        return make_context(start_t, goal_t)

    def _sample_one_plan(self, start_pos: list) -> np.ndarray:
        """Single goal-conditioned Diffusion sample, denormalised."""
        start_norm = normalize_grid(start_pos)
        goal_norm = self._goal_norm

        start_t = torch.tensor(start_norm, device=self.device, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal_norm, device=self.device, dtype=torch.float32).unsqueeze(0)

        sample_kw = dict(
            batch_size=1,
            horizon=self.horizon,
            state_dim=2,
            start=start_t,
            goal=goal_t,
        )
        if self.context_dim > 0:
            sample_kw["context"] = self._build_sample_context(start_pos, start_t, goal_t)

        with torch.no_grad():
            traj = self.diffusion.sample(**sample_kw, guidance_scale=self.guidance_scale)

        waypoints = _denorm(traj[0].cpu().numpy())
        waypoints[0] = np.array(start_pos, dtype=np.float32)
        waypoints[-1] = np.array(GOAL_POS, dtype=np.float32)
        return waypoints

    def _sample_waypoints(
        self,
        start_pos: list,
        *,
        laundry_phase: bool = True,
    ) -> np.ndarray:
        """
        Sample Diffusion plan; in laundry phase require route near a laundry pile.
        Falls back to multi-goal A* expert resampling if Diffusion fails.
        """
        for _ in range(self.plan_resample_attempts):
            candidate = self._sample_one_plan(start_pos)
            if laundry_phase and not self._plan_covers_laundry(candidate):
                candidate = self._inject_laundry_anchor(candidate, start_pos)
            if not self._plan_is_valid(candidate):
                continue
            if laundry_phase and not self._plan_covers_laundry(candidate):
                continue
            return candidate

        if laundry_phase:
            astar_wps = self._astar_plan_grid(start_pos, pet_aware=False)
            if astar_wps is not None:
                return astar_wps

        candidate = self._sample_one_plan(start_pos)
        if laundry_phase:
            candidate = self._inject_laundry_anchor(candidate, start_pos)
        return candidate

    def _replan_from(
        self,
        pos: list,
        *,
        laundry_phase: bool,
        reason: str = "manual",
        prefer_pet_aware_astar: bool = False,
    ) -> None:
        """Re-sample global waypoints from the robot's current cell."""
        if prefer_pet_aware_astar:
            astar_wps = self._astar_plan_grid(pos, pet_aware=True)
            if astar_wps is not None:
                self._waypoints = astar_wps
                self._reset_waypoint_tracking(pos)
                self._pos_stuck_steps = 0
                self._steps_since_replan = 0
                self._replan_count += 1
                return

        self._waypoints = self._sample_waypoints(pos, laundry_phase=laundry_phase)
        self._reset_waypoint_tracking(pos)
        self._pos_stuck_steps = 0
        self._steps_since_replan = 0
        self._replan_count += 1

    def _maybe_event_replan(
        self,
        row: int,
        col: int,
        *,
        strict_success: bool,
        laundry_ok: bool,
    ) -> tuple[bool, str]:
        """Phase 2 event-driven replan: stuck, PET proximity, path end."""
        if strict_success:
            return False, ""
        if self._steps_since_replan < REPLAN_COOLDOWN_STEPS:
            return False, ""

        if self._pos_stuck_steps >= STUCK_REPLAN_THRESHOLD:
            return True, "stuck"
        if self._on_or_near_pet(row, col):
            return True, "pet"
        if self._wp_idx >= self.horizon - 1:
            return True, "path_end"
        # Way 2: periodic replan so diffusion re-generates from current position
        if self._steps_since_replan >= PERIODIC_REPLAN_EVERY:
            return True, "periodic"
        return False, ""

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        base_obs, info = self.base_env.reset(seed=seed, options=options or {})
        self._laundry_cells = [tuple(c) for c in self.base_env.laundry_coords]
        start_pos = [base_obs[0], base_obs[1]]

        self._steps = 0
        self._episode_return = 0.0
        self._replan_count = 0
        self._pos_stuck_steps = 0
        self._steps_since_replan = REPLAN_COOLDOWN_STEPS
        self._prev_pos = (int(start_pos[0]), int(start_pos[1]))
        self._waypoints = self._sample_waypoints(start_pos, laundry_phase=True)
        self._reset_waypoint_tracking(start_pos)

        obs = self._make_obs(base_obs)
        info["waypoints"] = self._waypoints.copy()
        info["waypoint_idx"] = self._wp_idx
        info["horizon"] = self.horizon
        info["plan_hazard_frac"] = self._plan_hazard_fraction(self._waypoints)
        info["plan_covers_laundry"] = self._plan_covers_laundry(self._waypoints)
        info["pet_pos"] = self.base_env.pet_pos
        return obs, info

    def step(self, action: int):
        coins_before = sum(self.base_env.coins)
        base_obs, base_reward, done, info = self.base_env.step(action)
        self._steps += 1

        row, col = base_obs[0], base_obs[1]
        pos = np.array([row, col], dtype=np.float32)
        curr_pos = (int(row), int(col))

        if curr_pos == self._prev_pos:
            self._pos_stuck_steps += 1
        else:
            self._pos_stuck_steps = 0
        self._prev_pos = curr_pos
        self._steps_since_replan += 1

        goal = np.array(GOAL_POS, dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(pos - goal))
        strict_success = is_task_success(self.base_env, total_reward=None)
        laundry_ok = self._laundry_satisfied()

        goal_shaping = 0.0
        if laundry_ok:
            goal_shaping = GOAL_SHAPING_DELIVERY * (self._prev_dist_to_goal - dist_to_goal)
        self._prev_dist_to_goal = dist_to_goal

        wp = self._waypoints[self._wp_idx]
        dist = float(np.linalg.norm(pos - wp))
        shaping = self._prev_dist - dist
        self._prev_dist = dist

        wp_bonus_rate = WAYPOINT_BONUS_DELIVERY if laundry_ok else WAYPOINT_BONUS_LAUNDRY
        waypoint_reward = 0.0
        wp_advanced = False
        if dist <= self.reach_radius and self._wp_idx < self.horizon - 1:
            waypoint_reward += wp_bonus_rate
            waypoint_reward += SEQUENTIAL_PROGRESS_BONUS * self._wp_idx
            self._wp_idx += 1
            wp_advanced = True
            self._prev_dist = float(np.linalg.norm(pos - self._waypoints[self._wp_idx]))
        elif dist <= self.reach_radius and self._wp_idx == self.horizon - 1:
            waypoint_reward += wp_bonus_rate * 0.5

        if wp_advanced:
            self._stall_steps = 0
        else:
            self._stall_steps += 1
        stall_cost = 0.0
        if self._stall_steps >= self.stall_step_limit:
            stall_cost = self.stall_penalty
            self._stall_steps = 0

        coin_bonus = 0.0
        if base_reward == 10:
            coin_bonus = LAUNDRY_PICKUP_BONUS
        elif base_reward == 20:
            coin_bonus = 3.0

        laundry_shaping = 0.0
        curr_laundry_dist = self._nearest_laundry_dist(pos)
        if not laundry_ok:
            laundry_shaping = 0.8 * (self._prev_laundry_dist - curr_laundry_dist)
        self._prev_laundry_dist = curr_laundry_dist

        replanned = False
        replan_reason = ""
        coins_after = sum(self.base_env.coins)
        delivery_phase = laundry_ok

        if (
            coins_before < self.base_env.min_laundry_required
            and coins_after >= self.base_env.min_laundry_required
        ):
            self._replan_from(
                [row, col], laundry_phase=False, reason="laundry",
            )
            replanned = True
            replan_reason = "laundry"
        else:
            need_replan, event_reason = self._maybe_event_replan(
                row, col, strict_success=strict_success, laundry_ok=laundry_ok,
            )
            if need_replan:
                prefer_astar = event_reason in ("stuck", "pet")
                # periodic replanning uses diffusion (fresh trajectory from current pos)
                if event_reason == "periodic":
                    prefer_astar = False
                self._replan_from(
                    [row, col],
                    laundry_phase=not delivery_phase,
                    reason=event_reason,
                    prefer_pet_aware_astar=prefer_astar,
                )
                replanned = True
                replan_reason = event_reason

        goal_bonus = GOAL_COMPLETION_BONUS if strict_success else 0.0
        total_reward = (
            base_reward + waypoint_reward + coin_bonus + goal_bonus
            + self.distance_shaping_coef * shaping
            + GOAL_SHAPING_COEF * goal_shaping
            + laundry_shaping
            - stall_cost
        )
        self._episode_return += total_reward

        terminated = done or strict_success
        truncated = (not terminated) and (self._steps >= self.max_steps)
        if truncated and not strict_success:
            total_reward += TIMEOUT_PENALTY
            self._episode_return += TIMEOUT_PENALTY

        obs = self._make_obs(base_obs)
        info["waypoint_idx"] = self._wp_idx
        info["dist_to_waypoint"] = dist
        info["dist_to_goal"] = dist_to_goal
        info["waypoints_done"] = self._wp_idx
        info["goal_reached"] = strict_success
        info["task_success"] = strict_success
        info["strict_task_success"] = is_task_success(
            self.base_env, total_reward=self._episode_return,
        )
        info["laundry_satisfied"] = laundry_ok
        info["replanned"] = replanned
        info["replan_reason"] = replan_reason
        info["replanned_after_laundry"] = replan_reason == "laundry"
        info["replan_count"] = self._replan_count
        info["pos_stuck_steps"] = self._pos_stuck_steps
        info["stall_steps"] = self._stall_steps
        info["wp_advanced"] = wp_advanced
        info["pet_pos"] = self.base_env.pet_pos

        return obs, total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.base_env.render()

    def close(self):
        self.base_env.close()
