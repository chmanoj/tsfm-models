"""SegmentSpec (D9.2) — everything needed to compute the segment length ``S``
and to assemble the tokens later, *without* tokenizing at packing time.

A segment is one cropped sample. All channels of a sample share the same length
and forecast origin, so they receive the *identical* telescope (D5: all channels
get the same tokenization) — hence per-tier ``counts`` is one vector for the
whole segment. Channel-major layout (D8): within a segment, each channel's block
is ``[n_eff context tokens | horizon tokens]``; horizon tokens are query [MASK]
tokens for target channels and observed known-future-feature (KFF) tokens for
revealed feature channels (D6/D11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..constants import RoleFT, V


@dataclass
class SegmentSpec:
    # --- sample shape ---
    n_features: int          # native feature channels (features-first)
    n_targets: int           # native target channels
    t_raw: int               # raw series length

    # --- crop ---
    origin: int              # raw index one past last context step (forecast origin)
    p: int                   # horizon length in raw steps (variable-p, D6)
    q_tok: int               # horizon query tokens per target channel = ceil(p / p_out)
    p_out: int               # horizon output patch size (P_out)

    n_eff: int               # context tokens per channel
    counts: List[int]        # per-tier context-token counts (len V); same for all channels

    # --- D11 roles (post-augmentation), per channel (len C) ---
    role_ft: List[int]       # RoleFT per channel
    kff: List[bool]          # feature channel reveals its horizon (known-future feature)

    def __post_init__(self) -> None:
        assert len(self.counts) == V, self.counts
        assert len(self.role_ft) == self.C, (len(self.role_ft), self.C)
        assert len(self.kff) == self.C
        assert sum(self.counts) == self.n_eff, (self.counts, self.n_eff)

    @property
    def C(self) -> int:
        return self.n_features + self.n_targets

    @property
    def n_target_eff(self) -> int:
        return sum(1 for r in self.role_ft if r == int(RoleFT.TARGET))

    @property
    def n_kff(self) -> int:
        return sum(1 for f in self.kff if f)

    @property
    def Q(self) -> int:
        """Horizon query tokens in this segment (target channels only)."""
        return self.n_target_eff * self.q_tok

    @property
    def n_horizon_obs(self) -> int:
        """Observed known-future-feature horizon tokens (revealed features)."""
        return self.n_kff * self.q_tok

    def horizon_tokens(self, channel: int) -> int:
        """Horizon tokens attached to a given channel's block (channel-major)."""
        if self.role_ft[channel] == int(RoleFT.TARGET):
            return self.q_tok
        if self.kff[channel]:
            return self.q_tok
        return 0

    @property
    def S(self) -> int:
        """Exact segment length (tokens), computable from the spec alone (D9)."""
        return self.C * self.n_eff + self.Q + self.n_horizon_obs
