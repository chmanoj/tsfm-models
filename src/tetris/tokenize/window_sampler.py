"""Window sampler (walkthrough Stage 1) — origin/horizon sampling + integer budget
fields. Pure, per item, shape-only (no data values), so segments are packable
without tokenizing (D9). Emits a scalar ``SegmentSpec``.

Budget (walkthrough): ``Q = n_targets·⌈p/P_out⌉``, ``K = n_kff·⌈p/P_out⌉``,
``Q_total = Q+K``, ``n = ⌊(L − Q_total)/C⌋``. KFF is the horizon-side family with
queries, charged to ``Q_total`` (not to the per-channel context budget ``n``).

D11 note: only the KFF-reveal half of role augmentation is kept (each native
feature is revealed known-future w.p. ``kff_reveal_prob``); the target→feature
demotion half is deferred — it needs channel reordering, incompatible with the
scalar, features-first ``SegmentSpec`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import NamedTuple, Optional, Tuple

import numpy as np

from .. import telescope as TS
from ..constants import ChannelRole, PATCH
from .spec import SegmentSpec


@dataclass
class SamplerParams:
    l_pack: int
    p_out: int
    tier_prior: Tuple[int, ...] = (16, 16, 16, 16, 16, 8)  # D12 ratio prior
    q_tok_max: int = 4              # max horizon tokens per horizon channel (design point: 4)
    kff_reveal_prob: float = 0.0    # per native feature, reveal known-future (D11, partial)
    # G3.1 query-token budget: Q_total = n_horizon_channels·q_tok ≤ max_query_tokens.
    # cfg construction passes packing.max_query_tokens; the very large default means
    # "bounded only by L_pack" (the existing per-channel budget) so direct-construction
    # unit tests are unchanged. Never an off-switch — it is always a real budget.
    max_query_tokens: int = 1 << 30
    # D13 phase-2 ``auto_from_test_configs``: when set (non-empty), the horizon p is
    # drawn from this empirical set of GIFT-Eval test prediction-lengths (clamped to
    # the series + token budget) instead of the broad log-uniform-within-band default,
    # so phase-2 training crops match the test horizon marginal. None -> broad default.
    crop_horizons: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        assert self.p_out in PATCH, f"p_out {self.p_out} must be in the patch vocabulary {PATCH}"


class FixedWindow(NamedTuple):
    """Per-item deterministic crop intent (H1, noise-robustness Path B).

    A generator that bakes a noisy *context* of ``ctx`` raw steps followed by a
    clean (conditional-mean) *horizon* of ``horizon`` raw steps attaches this so
    the sampler crops at exactly ``origin = ctx, p = horizon`` instead of the
    random origin/horizon it draws for ordinary items — otherwise the random crop
    would slice the horizon back into the noisy region and the clean-GT property
    would be lost (see the H1 reconciliation block). ``horizon`` is still clamped
    to the per-pass token budget; clamping only *shortens* the horizon, which stays
    inside the clean region, so the property holds.
    """

    ctx: int
    horizon: int


def sample_window(n_features: int, n_targets: int, t_raw: int, params: SamplerParams,
                  rng, fixed: Optional[FixedWindow] = None) -> SegmentSpec:
    """Sample one crop spec from a sample's shape (walkthrough Stage 1).

    ``fixed`` (default ``None`` → unchanged random sampling) pins the crop to a
    baked ``(ctx, horizon)`` for noise-robustness items; every non-fixed path is
    byte-identical to before."""
    C = n_features + n_targets
    assert C >= 1 and n_targets >= 1 and t_raw >= 2

    # Per-channel roles (features-first): features [0,n_features), targets after.
    # KFF-reveal among native features (D11, partial; demotion deferred).
    channel_roles = [int(ChannelRole.FEATURE)] * n_features + [int(ChannelRole.TARGET)] * n_targets
    for c in range(n_features):
        if rng.random() < params.kff_reveal_prob:
            channel_roles[c] = int(ChannelRole.KFF)
    n_kff = sum(r == int(ChannelRole.KFF) for r in channel_roles)
    n_horizon_channels = n_targets + n_kff

    # Cap horizon so >=1 context token per channel fits and it fits the series.
    max_q_by_budget = max(1, (params.l_pack - C) // max(1, n_horizon_channels))
    max_q_by_series = max(1, (t_raw - 1) // params.p_out)
    # G3.1: also bound by the max-query-token budget so a training segment never
    # one-shots a huge horizon (Q_total = n_horizon_channels·q_tok ≤ max_query_tokens).
    max_q_by_cap = max(1, params.max_query_tokens // max(1, n_horizon_channels))
    q_tok_max = max(1, min(params.q_tok_max, max_q_by_budget, max_q_by_series, max_q_by_cap))

    if fixed is not None:
        # H1 Path B: deterministic crop at the baked (ctx, horizon). Clamp the
        # horizon to the series + per-pass token budget (only ever shortens it, so
        # it stays inside the clean region); origin is pinned below.
        p = max(1, min(int(fixed.horizon), t_raw - 1, q_tok_max * params.p_out))
        q_tok = ceil(p / params.p_out)
    elif params.crop_horizons:
        # D13 phase-2: draw a test-matched horizon, clamp to series + token budget.
        # The budget cap (q_tok_max·P_out) bounds the per-pass horizon exactly like
        # training always does — eval still rolls out the full test horizon (G3.1).
        p_target = int(params.crop_horizons[rng.integers(0, len(params.crop_horizons))])
        p = max(1, min(p_target, t_raw - 1, q_tok_max * params.p_out))
        q_tok = ceil(p / params.p_out)
    else:
        q_tok = int(rng.integers(1, q_tok_max + 1))
        # variable-p within the chosen query-token band (D6): p in ((q_tok-1)·P, q_tok·P]
        p_hi = min(q_tok * params.p_out, t_raw - 1)
        p_lo = min((q_tok - 1) * params.p_out + 1, p_hi)
        p = int(rng.integers(p_lo, p_hi + 1))
    assert ceil(p / params.p_out) == q_tok

    Q = n_targets * q_tok
    K = n_kff * q_tok
    Q_total = Q + K
    n = max(1, (params.l_pack - Q_total) // C)

    # Origin: pinned to the baked context length for fixed windows (clamped to a
    # valid position), else uniform over positions with >=1 context step and p
    # future steps (D5: every position is a forecast origin).
    if fixed is not None:
        origin = int(np.clip(fixed.ctx, 1, t_raw - p))
    else:
        origin = int(rng.integers(1, (t_raw - p) + 1))

    # Coverage budget from n, clipped to available history before the origin.
    prior = params.tier_prior
    t_cov = TS.coverage(TS.default_counts(n, prior))
    target_cov = min(t_cov, origin)
    n_eff = max(1, TS.tokens_for_coverage(target_cov, max_tokens=n, prior=prior))
    counts = TS.allocate(n_eff, target_cov, prior=prior)

    return SegmentSpec(
        C=C, n_features=n_features, n_targets=n_targets,
        origin=origin, p=p, channel_roles=channel_roles,
        Q=Q, K=K, Q_total=Q_total, n=n, counts=counts,
    )
