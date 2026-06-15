"""Variate identity embedding (D4, walkthrough Stage 6 'variate' part).

``variate_emb = random orthonormal ID[variate_uid] + scale-receipt MLP(a, σΔ)``.

The random ID is a *register*: early layers discover from content which channel
matters, later layers route by tag ("discover once, address thereafter"). It is
**resampled every sample** so IDs are usable only relationally (fixed seed at
inference). The basis is **generated eagerly outside the compiled forward** (QR)
and passed in — keeping the orthonormalization out of the graph (D14 no-recompile).

``variate_uid`` is buffer-unique per ``(sample, channel)`` (the collator rebases
it), so one orthonormal pool **per buffer** separates every variate in that buffer
(attention never crosses buffers). The scale-receipt MLP feeds the per-token
``(stats_a, stats_sigma)`` (D10 base scale) so the model sees each variate's scale.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def sample_orthonormal_basis(
    B: int,
    n_var: int,
    d_model: int,
    *,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-buffer random orthonormal ID pool ``[B, n_var, d]`` (D4).

    Rows are orthonormal when ``n_var <= d`` (rows of a random rotation, via QR);
    if a buffer needs more variates than ``d`` dimensions allow, the rows fall back
    to L2-normalized Gaussian (near-orthogonal in high ``d``). Eager — call once
    per step (resample) before the compiled forward; fix the generator at inference.
    """
    g = torch.randn(B, d_model, max(n_var, 1), device=device, generator=generator)
    if n_var <= d_model:
        q, _ = torch.linalg.qr(g)             # q: [B, d, n_var], orthonormal columns
        return q.transpose(1, 2).contiguous()  # [B, n_var, d], orthonormal rows
    rows = g.transpose(1, 2)                    # [B, n_var, d]
    return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class VariateID(nn.Module):
    """Random orthonormal ID (gathered) + scale-receipt MLP over ``(a, σΔ[0])``."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale_receipt = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(
        self,
        variate_uid: torch.Tensor,   # int [B, L]  (pad = -1)
        stats_a: torch.Tensor,       # float [B, L]
        stats_sigma: torch.Tensor,   # float [B, L]
        basis: torch.Tensor,         # float [B, n_var, d]  (eager, hoisted)
    ) -> torch.Tensor:
        B, L = variate_uid.shape
        uid = variate_uid.clamp_min(0).long()                       # pad -1 → row 0
        ids = torch.gather(basis, 1, uid.unsqueeze(-1).expand(B, L, self.d_model))
        receipt = self.scale_receipt(torch.stack([stats_a, stats_sigma], dim=-1))
        return ids + receipt                                        # [B, L, d]
