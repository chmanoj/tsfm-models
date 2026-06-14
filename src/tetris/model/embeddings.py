"""Token embedding — the 5-part sum (walkthrough Stage 6).

``tokens = content + time(t_center) + span(tier_id) + variate + role_ft(role_ft)``

``content`` (Stage 7 encoder output | learned [MASK]/[NA]) and ``variate`` (D4)
are computed by their own modules and passed in; this module owns the three small
learned parts. Position is carried **only** by ``t_center`` (D8: no buffer-index
positional encoding) — time uses a continuous Fourier embedding (own low-frequency
range, since ``t_center`` spans ±thousands of steps), the rest are lookup tables.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..constants import RoleFT, V
from .encoders import fourier_features


class TimeEmbedding(nn.Module):
    """Continuous embedding of ``t_center`` (float) → ``[..., d]``."""

    def __init__(self, d_model: int, n_fourier: int = 32):
        super().__init__()
        # Wavelengths ~6..6000 steps: capture both near-origin and deep-history time.
        freqs = torch.exp(torch.linspace(math.log(1e-3), math.log(1.0), n_fourier))
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(2 * n_fourier, d_model)

    def forward(self, t_center: torch.Tensor) -> torch.Tensor:
        return self.proj(fourier_features(t_center, self.freqs))


class TokenEmbedding(nn.Module):
    """Stage-6 sum of the five token-embedding parts."""

    def __init__(self, d_model: int, n_fourier_time: int = 32):
        super().__init__()
        self.time = TimeEmbedding(d_model, n_fourier_time)
        self.span = nn.Embedding(V, d_model)              # tier_id 0..5 (D2/D12)
        self.role_ft = nn.Embedding(len(RoleFT), d_model)  # FEATURE / TARGET (D11)

    def forward(
        self,
        batch,
        content: torch.Tensor,   # [B, L, d]  (Stage 7)
        variate: torch.Tensor,   # [B, L, d]  (D4)
    ) -> torch.Tensor:
        return (
            content
            + variate
            + self.time(batch.t_center)
            + self.span(batch.tier_id.long())
            + self.role_ft(batch.role_ft.long())
        )
