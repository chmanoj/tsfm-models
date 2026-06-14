"""SegmentSpec (walkthrough Stage 1).

A segment is one cropped sample. All channels share the same length and forecast
origin, so they receive the identical telescope (D5) — ``counts`` is one 6-int
vector for the whole segment. Per-channel role is carried explicitly as
``channel_roles`` (a length-C list of ``ChannelRole`` ∈ {FEATURE, TARGET, KFF});
this is CPU pack-time data and never reaches the GPU, so it does not affect the
static compiled shapes. KFF = a feature whose horizon is revealed as observed
tokens (D11); it is *not* a separate token role (see assemble).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..constants import ChannelRole, V


@dataclass
class SegmentSpec:
    C: int
    n_features: int            # native features (FEATURE + KFF channels)
    n_targets: int             # native targets (TARGET channels)
    origin: int                # raw index where context ends / horizon begins
    p: int                     # horizon length in raw steps (variable-p, D6)
    channel_roles: List[int]   # per channel: ChannelRole {FEATURE, TARGET, KFF}
    Q: int                     # query tokens = n_targets · ⌈p/P_out⌉
    K: int                     # KFF tokens   = n_kff    · ⌈p/P_out⌉
    Q_total: int               # Q + K
    n: int                     # per-channel context-token budget = ⌊(L − Q_total)/C⌋
    counts: List[int]          # per-tier context-token counts (len V); same for all channels

    def __post_init__(self) -> None:
        assert len(self.counts) == V, self.counts
        assert len(self.channel_roles) == self.C
        assert self.C == self.n_features + self.n_targets
        assert self.n_targets == sum(r == int(ChannelRole.TARGET) for r in self.channel_roles)
        assert self.n_kff <= self.n_features
        assert self.Q_total == self.Q + self.K
        assert sum(self.counts) <= self.n

    @property
    def n_kff(self) -> int:
        return sum(r == int(ChannelRole.KFF) for r in self.channel_roles)

    @property
    def n_eff(self) -> int:
        """Actual context tokens per channel (Σcounts ≤ budget n)."""
        return sum(self.counts)

    @property
    def q_tok(self) -> int:
        """Horizon tokens per horizon channel = ⌈p/P_out⌉ (uniform across channels)."""
        h = self.n_targets + self.n_kff
        return self.Q_total // h if h > 0 else 0

    def role_of(self, channel: int) -> int:
        return self.channel_roles[channel]

    def horizon_tokens(self, channel: int) -> int:
        """Horizon tokens attached to a channel's block (channel-major). Targets get
        query tokens, KFF features get observed KFF tokens, past-only features none."""
        r = self.channel_roles[channel]
        return self.q_tok if r in (int(ChannelRole.TARGET), int(ChannelRole.KFF)) else 0

    @property
    def S(self) -> int:
        """Exact segment length (tokens): channel-major context + all horizon tokens."""
        return self.C * self.n_eff + self.Q_total
