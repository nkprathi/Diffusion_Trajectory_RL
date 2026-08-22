"""
unet.py  —  Phase 2

Temporal 1-D U-Net for robot trajectory denoising.

Architecture:
    SinusoidalPosEmb  → converts integer timestep t to a dense embedding.
    ResidualBlock1D   → Conv1D → GroupNorm → Mish + FiLM timestep conditioning.
    TemporalUNet      → Encoder → Bottleneck → Decoder with skip connections.

Input / Output:  (B, H, D) noisy trajectory → (B, H, D) predicted noise ε.

No external dependencies beyond PyTorch (einops NOT required).

References:
    Ho et al. (2020) — DDPM — arXiv:2006.11239
    Janner et al. (2022) — Diffuser — arXiv:2205.09991
"""

from __future__ import annotations

import math
from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Fixed sinusoidal embedding for integer diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t : (B,) integer timestep tensor

        Returns:
            (B, dim) embedding
        """
        device = t.device
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=device) * -freq)
        emb = t.float()[:, None] * freq[None, :]      # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# 1-D Residual block with FiLM timestep conditioning
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """
    Layout:
        x → Conv1D → GroupNorm → Mish → [FiLM scale/shift] → Conv1D → + → out
        x ─────────────────────────────────────────────────────────────── (1×1)

    The timestep embedding is projected to (2 · out_channels) to produce
    per-channel scale γ and shift β (FiLM / AdaIN style conditioning).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int = 8,
    ):
        super().__init__()
        g_in  = self._safe_groups(in_channels,  groups)
        g_out = self._safe_groups(out_channels, groups)

        self.conv1 = nn.Conv1d(in_channels,  out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(g_out, out_channels)
        self.act1  = nn.Mish()

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(g_out, out_channels)
        self.act2  = nn.Mish()

        self.time_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_dim, out_channels * 2),
        )

        self.res_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    @staticmethod
    def _safe_groups(channels: int, groups: int) -> int:
        """Reduce groups until it divides channels evenly."""
        while channels % groups != 0 and groups > 1:
            groups //= 2
        return groups

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x     : (B, C_in, L)
            t_emb : (B, time_embed_dim)

        Returns:
            (B, C_out, L)
        """
        h = self.act1(self.norm1(self.conv1(x)))

        # FiLM conditioning
        scale_shift = self.time_proj(t_emb)          # (B, 2·C_out)
        scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, C_out)
        h = h * (scale.unsqueeze(-1) + 1.0) + shift.unsqueeze(-1)

        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# Temporal U-Net
# ---------------------------------------------------------------------------

class TemporalUNet(nn.Module):
    """
    U-Net for robot trajectory sequences.

    Input  (B, H, D)  → output  (B, H, D)  [predicted noise ε].

    Internally transposes to (B, D, H) for 1-D convolutions and back.

    Args:
        state_dim      : Dimension of each waypoint (2 for 2-D grid).
        horizon        : Number of waypoints per trajectory (H=64).
        channels       : Channel widths at each U-Net level.
        time_embed_dim : Width of the time embedding MLP.
        context_dim    : Dimension of optional conditioning vector (0 = none).
    """

    def __init__(
        self,
        state_dim: int = 2,
        horizon: int = 64,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        time_embed_dim: int = 128,
        context_dim: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon

        # ------------------------------------------------------------------
        # Time embedding: sinusoidal → MLP
        # ------------------------------------------------------------------
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Optional context (e.g. concatenated start+goal position)
        self.context_proj = (
            nn.Linear(context_dim, time_embed_dim)
            if context_dim > 0 else None
        )

        # ------------------------------------------------------------------
        # Encoder (downsampling)
        # ------------------------------------------------------------------
        in_ch = state_dim
        self.enc_blocks: List[nn.ModuleList] = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for out_ch in channels[:-1]:
            self.enc_blocks.append(nn.ModuleList([
                ResidualBlock1D(in_ch,   out_ch, time_embed_dim),
                ResidualBlock1D(out_ch,  out_ch, time_embed_dim),
            ]))
            self.downsamplers.append(nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        mid_ch = channels[-1]
        self.mid1 = ResidualBlock1D(in_ch,   mid_ch, time_embed_dim)
        self.mid2 = ResidualBlock1D(mid_ch,  mid_ch, time_embed_dim)

        # ------------------------------------------------------------------
        # Decoder (upsampling)
        # ------------------------------------------------------------------
        self.upsamplers   = nn.ModuleList()
        self.dec_blocks   = nn.ModuleList()

        for skip_ch, out_ch in zip(reversed(channels[:-1]), reversed(channels[:-1])):
            self.upsamplers.append(
                nn.ConvTranspose1d(mid_ch, mid_ch, 4, stride=2, padding=1)
            )
            self.dec_blocks.append(nn.ModuleList([
                ResidualBlock1D(mid_ch + skip_ch, out_ch, time_embed_dim),
                ResidualBlock1D(out_ch,           out_ch, time_embed_dim),
            ]))
            mid_ch = out_ch

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------
        g = ResidualBlock1D._safe_groups(mid_ch, 8)
        self.out_norm = nn.GroupNorm(g, mid_ch)
        self.out_act  = nn.Mish()
        self.out_conv = nn.Conv1d(mid_ch, state_dim, 1)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x       : Noisy trajectory  (B, H, D)
            t       : Diffusion timestep indices  (B,)
            context : Optional conditioning vector  (B, context_dim)

        Returns:
            Predicted noise  (B, H, D)
        """
        # (B, H, D) → (B, D, H) for 1-D convolutions
        h = x.permute(0, 2, 1)

        # Time embedding
        t_emb = self.time_mlp(t)                        # (B, time_embed_dim)
        if self.context_proj is not None and context is not None:
            t_emb = t_emb + self.context_proj(context)

        # Encoder
        skips: list[Tensor] = []
        for (b1, b2), down in zip(self.enc_blocks, self.downsamplers):
            h = b1(h, t_emb)
            h = b2(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Decoder
        for up, (b1, b2), skip in zip(self.upsamplers, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = h[..., : skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = b1(h, t_emb)
            h = b2(h, t_emb)

        # Output
        h = self.out_conv(self.out_act(self.out_norm(h)))
        # (B, D, H) → (B, H, D)
        return h.permute(0, 2, 1)
