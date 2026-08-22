"""Standard diffusion planner checkpoint filenames."""

from __future__ import annotations

import os

DIFFUSION_BEST = "diffusion_best.pt"
DIFFUSION_FINAL = "diffusion_final.pt"
DIFFUSION_RL_FINETUNED = "diffusion_rl_finetuned.pt"
DIFFUSION_TASKCOND = "diffusion_taskcond.pt"

_LEGACY_MAP = {
    DIFFUSION_BEST: "ddpm_best.pt",
    DIFFUSION_FINAL: "ddpm_final.pt",
    DIFFUSION_RL_FINETUNED: "ddpm_rl_finetuned.pt",
    DIFFUSION_TASKCOND: "ddpm_taskcond.pt",
}


def resolve_checkpoint(checkpoint_dir: str, name: str) -> str:
    """Return path to checkpoint, falling back to legacy ddpm_*.pt if needed."""
    primary = os.path.join(checkpoint_dir, name)
    if os.path.isfile(primary):
        return primary
    legacy = _LEGACY_MAP.get(name)
    if legacy:
        fallback = os.path.join(checkpoint_dir, legacy)
        if os.path.isfile(fallback):
            return fallback
    return primary
