"""Window sampler (D9.2) — origin/horizon sampling, per-tier counts, role
augmentation. Stateless given an RNG; works on *shape only* (no data values), so
segments are packable without tokenizing (D9). Lives in the reservoir layer.

Budget logic (D9.2): horizon query tokens per target channel ``q_tok`` are drawn
first; the per-channel context allowance is ``n = ⌊(L_pack − NC)/C⌋`` where
``NC`` is all non-context (query + KFF-horizon) tokens; the raw coverage is the
telescope inverse of ``n``; the forecast origin is sampled uniformly over valid
positions; the context spans up to that coverage ending at the origin. Long
series therefore yield many crops across all their regimes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import telescope as TS
from ..constants import PATCH, RoleFT
from .spec import SegmentSpec


@dataclass
class SamplerParams:
    l_pack: int
    p_out: int
    tier_prior: tuple = tuple(int(w) for w in (16, 16, 16, 16, 16, 8))  # D12 ratio prior
    q_tok_max: int = 4              # max horizon query tokens per target (D12 design point: 4)
    role_aug_prob: float = 0.2      # crop-level prob q of applying role augmentation (D11)
    demote_prob: float = 0.5        # per-channel demote-to-feature prob
    reveal_prob: float = 0.5        # per demoted feature, reveal-horizon (KFF) prob

    def __post_init__(self) -> None:
        assert self.p_out in PATCH, f"p_out {self.p_out} must be in the patch vocabulary {PATCH}"


def _assign_roles(rng, n_features, n_targets, params: SamplerParams):
    """D11 light role augmentation. Native features stay past-only features;
    augmentation (crop-level prob q) demotes target channels to features (keeping
    ≥1 target) and reveals some demoted features as KFF."""
    C = n_features + n_targets
    role = [int(RoleFT.FEATURE)] * n_features + [int(RoleFT.TARGET)] * n_targets
    kff = [False] * C

    if n_targets > 0 and rng.random() < params.role_aug_prob:
        target_idx = list(range(n_features, C))
        n_keep_min = 1
        demotable = target_idx[:]  # all current targets are candidates
        # keep at least one target
        for c in demotable:
            remaining_targets = sum(1 for r in role if r == int(RoleFT.TARGET))
            if remaining_targets <= n_keep_min:
                break
            if rng.random() < params.demote_prob:
                role[c] = int(RoleFT.FEATURE)
                if rng.random() < params.reveal_prob:
                    kff[c] = True
    return role, kff


def sample_window(n_features: int, n_targets: int, t_raw: int, params: SamplerParams,
                  rng) -> SegmentSpec:
    """Sample one crop spec from a sample's shape (D9.2)."""
    C = n_features + n_targets
    assert C >= 1 and t_raw >= 2

    role_ft, kff = _assign_roles(rng, n_features, n_targets, params)
    n_target_eff = sum(1 for r in role_ft if r == int(RoleFT.TARGET))
    n_kff = sum(1 for f in kff if f)
    n_horizon_channels = n_target_eff + n_kff

    # Cap horizon so at least one context token per channel fits:
    #   C * 1 + n_horizon_channels * q_tok * p_out_tokens <= L_pack
    # (each horizon token corresponds to one p_out patch).
    max_q_by_budget = max(1, (params.l_pack - C) // max(1, n_horizon_channels))
    # Cap horizon so it fits in the raw series (need >=1 context step).
    max_q_by_series = max(1, (t_raw - 1) // params.p_out)
    q_tok_max = max(1, min(params.q_tok_max, max_q_by_budget, max_q_by_series))

    q_tok = int(rng.integers(1, q_tok_max + 1))
    # variable-p within the chosen query-token band (D6): p in ((q_tok-1)*P, q_tok*P]
    p_hi = min(q_tok * params.p_out, t_raw - 1)
    p_lo = min((q_tok - 1) * params.p_out + 1, p_hi)
    p = int(rng.integers(p_lo, p_hi + 1))

    # Non-context tokens and the per-channel context budget.
    nc = n_horizon_channels * q_tok
    n = max(1, (params.l_pack - nc) // C)

    # Raw coverage targeted by the budget, then clip to available history.
    prior = params.tier_prior
    t_cov = TS.coverage(TS.default_counts(n, prior))

    # Uniform random origin over valid positions: need >=1 context step and p future steps.
    origin_hi = t_raw - p
    origin_lo = 1
    assert origin_hi >= origin_lo, (t_raw, p)
    origin = int(rng.integers(origin_lo, origin_hi + 1))

    avail = origin  # context steps available before the origin
    target_cov = min(t_cov, avail)
    n_eff = TS.tokens_for_coverage(target_cov, max_tokens=n, prior=prior)
    n_eff = max(1, n_eff)
    counts = TS.allocate(n_eff, target_cov, prior=prior)
    # allocate preserves the budget exactly; keep n_eff in sync with sum(counts).
    n_eff = sum(counts)

    return SegmentSpec(
        n_features=n_features, n_targets=n_targets, t_raw=t_raw,
        origin=origin, p=p, q_tok=q_tok, p_out=params.p_out,
        n_eff=n_eff, counts=counts, role_ft=role_ft, kff=kff,
    )
