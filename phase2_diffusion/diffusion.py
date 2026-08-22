"""
diffusion.py  —  Phase 2

GaussianDiffusion: wraps TemporalUNet with Diffusion training loss and
reverse-diffusion sampling (with optional endpoint pinning).

References:
    Ho et al. (2020) — DDPM — arXiv:2006.11239
    Carvalho et al. (2023) — Motion Planning Diffusion — arXiv:2308.01557
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .noise_schedule import NoiseSchedule
from .unet import TemporalUNet


class GaussianDiffusion:
    """
    Diffusion training + sampling for robot trajectories.

    Not an nn.Module — the U-Net is the only learnable component.
    """

    def __init__(
        self,
        model: TemporalUNet,
        schedule: NoiseSchedule,
        device: str = "cpu",
        clip_denoised: bool = True,
    ):
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device
        self.clip_denoised = clip_denoised

    # ------------------------------------------------------------------
    # Training objective  L = E[||ε − ε_θ(τ_t, t)||²]
    # ------------------------------------------------------------------

    def training_loss(
        self,
        x0: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Standard DDPM noise-prediction MSE loss.

        Args:
            x0      : Clean trajectories  (B, H, D)
            context : Optional conditioning vector  (B, ctx_dim)

        Returns:
            Scalar loss.
        """
        B = x0.shape[0]
        x0 = x0.to(self.device)

        # Random timestep for each sample in batch
        t = torch.randint(0, self.schedule.T, (B,), device=self.device)

        # Sample Gaussian noise ε ~ N(0, I)
        noise = torch.randn_like(x0)

        # Forward process: τ_t = √ᾱ_t · τ_0 + √(1-ᾱ_t) · ε
        x_t = self.schedule.q_sample(x0, t, noise)

        # U-Net predicts ε
        if context is not None:
            context = context.to(self.device)
        eps_pred = self.model(x_t, t, context)

        return F.mse_loss(eps_pred, noise)

    def validation_loss(
        self,
        x0: Tensor,
        context: Optional[Tensor] = None,
        *,
        seed: int = 0,
        mc_samples: int = 8,
    ) -> float:
        """
        Low-variance validation estimate of the noise-prediction MSE.

        Averages over ``mc_samples`` fixed (t, ε) draws derived from ``seed`` so
        the same checkpoint always gets the same score and epoch-to-epoch
        changes reflect model improvement, not RNG luck.
        """
        B = x0.shape[0]
        x0 = x0.to(self.device)
        if context is not None:
            context = context.to(self.device)

        total = 0.0
        for mc in range(mc_samples):
            rng = np.random.default_rng(seed + mc)
            t = torch.tensor(
                rng.integers(0, self.schedule.T, size=B),
                device=self.device,
                dtype=torch.long,
            )
            noise = torch.from_numpy(
                rng.standard_normal(x0.shape).astype(np.float32)
            ).to(device=self.device)
            x_t = self.schedule.q_sample(x0, t, noise)
            eps_pred = self.model(x_t, t, context)
            total += F.mse_loss(eps_pred, noise).item()
        return total / mc_samples

    # ------------------------------------------------------------------
    # Reverse diffusion sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        horizon: int,
        state_dim: int,
        start: Optional[Tensor] = None,
        goal: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        return_chain: bool = False,
    ) -> Tensor:
        """
        Generate trajectories by running the reverse diffusion loop
        from pure Gaussian noise to a clean trajectory.

        Args:
            batch_size     : Number of trajectories to generate.
            horizon        : Number of waypoints H.
            state_dim      : Waypoint dimension D.
            start          : Optional start position (D,) — pinned at index 0.
            goal           : Optional goal position (D,) — pinned at index -1.
            context        : Optional conditioning (B, ctx_dim).
            guidance_scale : Classifier-free guidance scale (1.0 = no guidance).
                             >1.0 runs a second unconditional pass per step and
                             amplifies the conditional signal.
            temperature    : Noise scale during reverse diffusion (1.0 = standard).
                             >1.0 produces more diverse / spread-out trajectories.
            return_chain   : If True, also return the denoising chain.

        Returns:
            τ₀ : Generated trajectories  (B, H, D)
        """
        shape = (batch_size, horizon, state_dim)
        tau = torch.randn(shape, device=self.device) * temperature

        # Pin start and goal endpoints
        if start is not None:
            tau[:, 0] = start.to(self.device)
        if goal is not None:
            tau[:, -1] = goal.to(self.device)

        chain = [tau.clone()] if return_chain else None

        use_cfg = guidance_scale > 1.0 and context is not None

        for t_val in reversed(range(self.schedule.T)):
            t_batch = torch.full(
                (batch_size,), t_val, device=self.device, dtype=torch.long
            )

            if use_cfg:
                # Two forward passes: conditional and unconditional (context=None)
                # eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
                eps_cond   = self.model(tau, t_batch, context)
                eps_uncond = self.model(tau, t_batch, None)
                eps_pred   = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = self.model(tau, t_batch, context)

            mu_t = self.schedule.ddpm_mean(tau, t_batch, eps_pred)

            if t_val > 0:
                var = self.schedule.posterior_variance[t_val].item()
                tau = mu_t + (var ** 0.5) * torch.randn_like(tau)
            else:
                tau = mu_t

            # Re-pin endpoints every step to prevent drift
            if start is not None:
                tau[:, 0] = start.to(self.device)
            if goal is not None:
                tau[:, -1] = goal.to(self.device)

            if return_chain:
                chain.append(tau.clone())

        if self.clip_denoised:
            tau = tau.clamp(-1.5, 1.5)

        if return_chain:
            return tau, chain
        return tau
