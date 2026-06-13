"""Backend switch — the single seam where compile/FlexAttention is opt-in (D14).

- CUDA path: FlexAttention + ``torch.compile`` (the real D14 target). The D9 mask
  is a ``BlockMask`` (``mask_mod``); ``score_mod`` is the ready socket for the A3
  bias ladder and the phase-2 NSA migration.
- Mac (MPS/CPU) path: ``F.scaled_dot_product_attention`` with a materialized
  ``[L, L]`` bool mask, eager.

The two paths must produce numerically equal masking/attention on the same
inputs (tested in S5). ``attend`` here is the low-level QKV op; mask construction
lives in ``masks.py`` (S5). ``score_mod`` is accepted but unused on the SDPA path
in v1 (A3 ladder is deferred).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec. ``auto`` prefers CUDA, then MPS, then CPU."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def backend_kind(device: torch.device) -> str:
    """``flex`` on CUDA (FlexAttention + compile), else ``sdpa`` (eager)."""
    return "flex" if device.type == "cuda" else "sdpa"


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask=None,
    score_mod=None,
    *,
    kind: Optional[str] = None,
) -> torch.Tensor:
    """Backend-routed scaled-dot-product attention.

    q, k, v: ``[B, H, L, d_head]``.

    - ``attn_mask`` on the SDPA path is a boolean tensor broadcastable to
      ``[B, H, L, L]`` where ``True`` means the (query, key) pair participates,
      or ``None`` for full attention. On the Flex path it is a ``BlockMask``.
    - ``score_mod`` is the A3-ladder/NSA socket; honored on the Flex path,
      ignored on the SDPA path in v1.
    """
    if kind is None:
        kind = "flex" if q.is_cuda else "sdpa"

    if kind == "flex":
        # Lazy import: FlexAttention is only available/relevant on the CUDA path.
        from torch.nn.attention.flex_attention import flex_attention

        return flex_attention(q, k, v, block_mask=attn_mask, score_mod=score_mod)

    # SDPA / eager path (Mac, CPU). Boolean attn_mask: True = participate.
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
