"""Global constants, enums, and dtypes shared across TETRIS modules.

Source of truth: docs/tetris/tetris_decision_log.html.
- PATCH vocabulary is fixed by D12 (six tiers, no per-step tier).
- Token *geometry* lives only in side tensors (D8 hard rule: no buffer-index
  positional encoding anywhere); the enums below tag those side tensors.
"""

from __future__ import annotations

import enum

# --- D12: fixed telescope patch vocabulary (six tiers) ----------------------
# The vocabulary is fixed so the six per-tier encoders (D3) are shared across
# all samples; only per-tier token *counts* vary (counts are data, not shape).
PATCH: tuple[int, ...] = (4, 8, 16, 64, 256, 512)
V: int = len(PATCH)  # number of telescope tiers / per-tier encoders / per-tier sigma

# --- D9 packing sentinels ---------------------------------------------------
PAD_SAMPLE_ID: int = -1  # side-tensor sample_id for padding positions


class ContentState(enum.IntEnum):
    """D7: three content states per token position. MASK and NA must never be
    conflated — they carry opposite semantics."""

    OBSERVED = 0  # tier encoder over real values
    MASK = 1      # learned query embedding: "value exists in the future, predict it"
    NA = 2        # learned missing embedding: "no observation here"


class Role(enum.IntEnum):
    """D6/D9: context vs horizon-query token."""

    CTX = 0
    QRY = 1


class RoleFT(enum.IntEnum):
    """D11: feature vs target. Two states suffice — a known-future feature is an
    observed CTX token at a horizon timestamp, distinguished by its time embedding."""

    FEATURE = 0
    TARGET = 1
