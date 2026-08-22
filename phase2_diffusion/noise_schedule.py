"""
noise_schedule.py  —  Phase 2

DDPM noise schedules: linear, cosine, exponential.
Provides forward process q_sample and reverse process helpers.

References:
    Ho et al. (2020) — DDPM — arXiv:2006.11239
    Nichol & Dhariwal (2021) — Improved DDPM — cosine schedule
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor


class NoiseSchedule:
    """
    Manages all DDPM noise coefficients.

    Key quantities (all shape (T,)):
        betas           : β_t  — noise added at each step
        alphas          : α_t = 1 - β_t
        alpha_bars      : ᾱ_t = Π αᵢ  (cumulative product)
        sqrt_alpha_bars : √ᾱ_t
        sqrt_1m_ab      : √(1 - ᾱ_t)
    """

    def __init__(
        self,
        T: int = 100,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.T = T
        self.device = device

        betas = self._make_betas(schedule, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)
        self.alpha_bars_prev = alpha_bars_prev.to(device)

        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).to(device)
        self.sqrt_1m_ab = torch.sqrt(1.0 - alpha_bars).to(device)
        self.sqrt_recip_alphas = torch.rsqrt(alphas).to(device)

        # Posterior variance: σ²_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        post_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars + 1e-8)
        self.posterior_variance = post_var.to(device)

    # ------------------------------------------------------------------
    # Schedule builders
    # ------------------------------------------------------------------

    def _make_betas(self, schedule: str, beta_start: float, beta_end: float) -> Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, self.T)
        elif schedule == "cosine":
            s = 0.008
            x = torch.linspace(0, self.T, self.T + 1)
            f = torch.cos(((x / self.T) + s) / (1 + s) * math.pi / 2) ** 2
            ab = f / f[0]
            betas = 1 - ab[1:] / ab[:-1]
            return betas.clamp(1e-5, 0.999)
        elif schedule == "exponential":
            return torch.exp(torch.linspace(math.log(beta_start), math.log(beta_end), self.T))
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

    # ------------------------------------------------------------------
    # Index gather helper
    # ------------------------------------------------------------------

    def _gather(self, coeff: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
        """Gather coeff[t] and broadcast to shape."""
        out = coeff.gather(0, t.long().to(coeff.device))
        while out.dim() < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    # ------------------------------------------------------------------
    # Forward process  q(τ_t | τ_0)  — reparameterisation trick
    # ------------------------------------------------------------------

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """
        τ_t = √ᾱ_t · τ_0 + √(1 - ᾱ_t) · ε

        Args:
            x0    : Clean trajectory  (B, H, D)
            t     : Timestep indices  (B,)
            noise : Pre-sampled noise or None

        Returns:
            Noisy trajectory (B, H, D)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self._gather(self.sqrt_alpha_bars, t, x0.shape)
        s1 = self._gather(self.sqrt_1m_ab, t, x0.shape)
        return sa * x0 + s1 * noise

    # ------------------------------------------------------------------
    # Reverse process helpers
    # ------------------------------------------------------------------

    def ddpm_mean(self, xt: Tensor, t: Tensor, eps_pred: Tensor) -> Tensor:
        """
        Posterior mean:  μ_t = (1/√α_t) · (x_t − (β_t / √(1 − ᾱ_t)) · ε̂)
        """
        coeff = self._gather(self.betas / self.sqrt_1m_ab, t, xt.shape)
        recip = self._gather(self.sqrt_recip_alphas, t, xt.shape)
        return recip * (xt - coeff * eps_pred)
