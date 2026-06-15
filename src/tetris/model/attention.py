"""Multi-head attention — backend-routed, mask as a parameter (D9/D14, Stage 8).

QKV projection → ``[B, H, L, d_head]`` → :func:`tetris.backend.attend` (FlexAttention
on CUDA, SDPA on Mac/CPU) → output projection. The attention **mask is a
parameter** (``block_mask``): a per-layer schedule (the deferred A3 ladder) is then
an arg change, not a refactor. ``score_mod`` is the same deferred socket, passed
through to the backend (ignored on the SDPA path in v1).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..backend import attend


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, (d_model, n_heads)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,            # [B, L, d]
        block_mask=None,            # BlockMask (flex) | bool [B,1,L,L] (sdpa) | None
        *,
        kind: Optional[str] = None,
        score_mod=None,             # deferred A3/NSA socket
    ) -> torch.Tensor:
        B, L, d = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)        # each [B, H, L, d_head]
        out = attend(q, k, v, attn_mask=block_mask, score_mod=score_mod, kind=kind)
        out = out.transpose(1, 2).reshape(B, L, d)  # [B, L, d]
        return self.proj(out)
