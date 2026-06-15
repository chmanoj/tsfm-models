"""Pre-norm transformer block (attention + FFN), Stage 8.

Shape-invariant ``[B, L, d] → [B, L, d]``; the Stage-5 ``block_mask`` is passed
through (depth-invariant in v1). No positional encoding is added here — position
already lives in the token embedding (D8).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(
        self, x: torch.Tensor, block_mask=None, *, kind: Optional[str] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), block_mask, kind=kind)
        x = x + self.ffn(self.norm2(x))
        return x
