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
    """D7: selects what fills the token's *content slot*. Not an added embedding —
    it picks among {tier-encoder output, learned [MASK] vector, learned [NA] vector}.
    MASK and NA carry opposite semantics and must never be conflated."""

    OBSERVED = 0  # content slot = tier encoder over the real (value, observed) window
    MASK = 1      # learned query embedding: "value exists in the future, predict it"
    NA = 2        # learned missing embedding: "no observation here"


class Role(enum.IntEnum):
    """D6/D9: context vs horizon-query token. Drives the attention mask. A
    known-future feature (KFF) is *not* a separate role — it is an observed CTX
    token at a horizon timestamp (t_center > 0), distinguished by its time
    embedding, never by a role tag (D11)."""

    CTX = 0
    QRY = 1


class RoleFT(enum.IntEnum):
    """D11: the feature/target role *embedding* added to every token. Two states;
    KFF channels' tokens are FEATURE (their horizon-ness comes from t_center > 0)."""

    FEATURE = 0
    TARGET = 1


class ChannelRole(enum.IntEnum):
    """Per-channel designation carried in SegmentSpec (CPU pack-time). A KFF
    channel is a feature whose horizon is revealed as observed tokens (D11).
    Token-level RoleFT maps KFF → FEATURE."""

    FEATURE = 0
    TARGET = 1
    KFF = 2
