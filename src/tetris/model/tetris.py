"""TETRIS model — full forward (S7): embed → encode → backbone → heads.

Stage order (walkthrough): Stage 7 encoders fill the content slot; Stage 6 sums
the five embedding parts; the Stage-5 mask is built once per batch (backend-routed,
depth-invariant) and reused by every Stage-8 block; Stage-9 heads run dense over
all ``L``. Loss/inversion live in S8 — this returns raw head outputs.

The variate basis (D4) is sampled eagerly here when not supplied, but the arg
exists so a training step can **hoist** the QR out of the compiled forward and
resample it per step (fixed seed at inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from ..backend import backend_kind
from ..config import Config, resolved_encoder_cap
from ..masks import build_block_mask, build_sdpa_mask
from .blocks import Block
from .embeddings import TokenEmbedding
from .encoders import TierEncoders
from .heads import Heads
from .variate_id import VariateID, sample_orthonormal_basis


@dataclass
class ModelOutput:
    horizon: torch.Tensor          # [B, L, P_out]
    aux: List[torch.Tensor]        # 6 × [B, L, P_k]


class Tetris(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        d = m.d_model
        self.d_model = d
        self.encoders = TierEncoders(d, resolved_encoder_cap(cfg))
        self.variate_id = VariateID(d)
        self.embed = TokenEmbedding(d)
        self.blocks = nn.ModuleList(Block(d, m.n_heads) for _ in range(m.n_layers))
        self.heads = Heads(d, m.out_patch)

    def forward(
        self,
        batch,
        variate_basis: Optional[torch.Tensor] = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> ModelOutput:
        device = batch.sample_id.device
        kind = backend_kind(device)
        B, L = batch.sample_id.shape

        content = self.encoders(batch)
        if variate_basis is None:
            n_var = max(int(batch.variate_uid.max().item()) + 1, 1)
            variate_basis = sample_orthonormal_basis(
                B, n_var, self.d_model, device=device, generator=generator
            )
        variate = self.variate_id(
            batch.variate_uid, batch.stats_a, batch.stats_sigma, variate_basis
        )
        tokens = self.embed(batch, content, variate)

        if kind == "flex":
            mask = build_block_mask(batch.sample_id, batch.role, batch.t_center)
        else:
            mask = build_sdpa_mask(batch.sample_id, batch.role, batch.t_center)

        h = tokens
        for block in self.blocks:
            h = block(h, mask, kind=kind)

        horizon, aux = self.heads(h)
        return ModelOutput(horizon=horizon, aux=aux)
