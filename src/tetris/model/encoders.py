"""Per-tier encoders + index-routed dispatch (D3, walkthrough Stage 7).

Six encoders, one per patch width ``P_k`` in :data:`PATCH`. Each maps a token's
``[P_k, 2]`` window — channel 0 = normalized value, channel 1 = observed-indicator
(D7) — to one ``D``-vector. Per D3 the value is lifted by **Fourier number
features** (a raw scalar through ``Linear(1, d)`` has poor spectral properties),
concatenated with the observed bit, flattened over the window, and run through a
2-layer MLP. Uniform recipe across tiers (only ``P_k`` differs).

Dispatch is **index-routed at static ``ENCODER_CAP``** (D14: one compile graph, no
dynamic shapes). For each tier ``k`` we form a fixed ``[CAP]`` slot list of the
``content_state==OBSERVED & tier_id==k`` tokens (context + KFF), gather their
windows from the flat ``norm_values`` store via ``raw_start``, encode, and scatter
back. The routing uses a cumsum-rank + ``scatter_``/``index_add_`` pattern so every
op is fixed-shape (no data-dependent sizes). ``MASK``/``NA`` slots receive learned
``[MASK]``/``[NA]`` vectors instead of an encoder output (selected by
``content_state``); pad (``sample_id==-1``) is ``NA`` and inert.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..constants import ContentState, PATCH


def fourier_features(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Lift scalars ``x[...]`` to ``[..., 2*len(freqs)]`` via ``[sin(xω), cos(xω)]``."""
    ang = x.unsqueeze(-1) * freqs  # [..., F]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TierEncoder(nn.Module):
    """One tier's window encoder: ``[CAP, P_k, 2] → [CAP, D]`` (D3)."""

    def __init__(self, p_k: int, d_model: int, n_fourier: int = 16):
        super().__init__()
        self.p_k = p_k
        freqs = torch.exp(torch.linspace(math.log(0.25), math.log(64.0), n_fourier))
        self.register_buffer("freqs", freqs, persistent=False)
        in_dim = p_k * (2 * n_fourier + 1)  # per step: Fourier(value) + observed bit
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, win: torch.Tensor) -> torch.Tensor:
        value, observed = win[..., 0], win[..., 1]            # [CAP, P_k]
        feats = torch.cat(
            [fourier_features(value, self.freqs), observed.unsqueeze(-1)], dim=-1
        )                                                      # [CAP, P_k, 2F+1]
        return self.net(feats.reshape(feats.shape[0], -1))     # [CAP, D]


class TierEncoders(nn.Module):
    """All six encoders + the static index-routed dispatch (Stage 7)."""

    def __init__(self, d_model: int, encoder_cap: int, n_fourier: int = 16):
        super().__init__()
        self.d_model = d_model
        self.cap = encoder_cap
        self.encoders = nn.ModuleList(
            TierEncoder(p_k, d_model, n_fourier) for p_k in PATCH
        )
        self.mask_vec = nn.Parameter(torch.randn(d_model) * 0.02)  # learned [MASK] (D7)
        self.na_vec = nn.Parameter(torch.randn(d_model) * 0.02)    # learned [NA]  (D7)

    def forward(self, batch) -> torch.Tensor:
        """Batch → content slot ``[B, L, D]`` (encoder output | [MASK] | [NA])."""
        B, L = batch.tier_id.shape
        R = batch.norm_values.shape[1]
        N, CAP = B * L, self.cap
        device = batch.tier_id.device

        tier_flat = batch.tier_id.reshape(-1).long()
        cs_flat = batch.content_state.reshape(-1).long()
        rs_flat = batch.raw_start.reshape(-1).long()
        buf_id = torch.arange(B, device=device).view(B, 1).expand(B, L).reshape(-1)
        slot_idx = torch.arange(N, device=device)
        nv_flat = batch.norm_values.reshape(-1)
        obs_flat = batch.observed.reshape(-1).to(nv_flat.dtype)

        content_flat = torch.zeros(N, self.d_model, device=device, dtype=nv_flat.dtype)
        observed_state = int(ContentState.OBSERVED)

        for k, encoder in enumerate(self.encoders):
            p_k = PATCH[k]
            routed = (cs_flat == observed_state) & (tier_flat == k)      # [N]
            rank = torch.cumsum(routed.long(), 0) - 1                    # [N]
            in_cap = routed & (rank >= 0) & (rank < CAP)
            # Scatter each routed slot to its rank in a [CAP+1] buffer (trash = CAP).
            dest = torch.where(in_cap, rank, torch.full_like(rank, CAP))
            idxbuf = torch.zeros(CAP + 1, dtype=torch.long, device=device)
            idxbuf.scatter_(0, dest, slot_idx)
            idx_k = idxbuf[:CAP]                                         # [CAP] slot ids
            validbuf = torch.zeros(CAP + 1, dtype=torch.bool, device=device)
            validbuf.scatter_(0, dest, in_cap)
            valid_k = validbuf[:CAP]                                     # [CAP] real rows

            # Gather each selected token's P_k window from the flat store.
            base = buf_id[idx_k] * R + rs_flat[idx_k]                    # [CAP]
            offs = torch.arange(p_k, device=device)
            g = (base.unsqueeze(1) + offs.unsqueeze(0)).clamp_(0, B * R - 1)  # [CAP, P_k]
            win = torch.stack([nv_flat[g], obs_flat[g]], dim=-1)        # [CAP, P_k, 2]

            enc = encoder(win)                                          # [CAP, D]
            enc = torch.where(valid_k.unsqueeze(1), enc, torch.zeros_like(enc))
            content_flat.index_add_(0, idx_k, enc)  # sentinel rows (slot 0) add 0

        content = content_flat.view(B, L, self.d_model)
        is_mask = (batch.content_state == int(ContentState.MASK)).unsqueeze(-1)
        is_na = (batch.content_state == int(ContentState.NA)).unsqueeze(-1)
        content = torch.where(is_mask, self.mask_vec, content)
        content = torch.where(is_na, self.na_vec, content)
        return content
