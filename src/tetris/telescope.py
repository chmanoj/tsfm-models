"""D2 / D12 — telescoping multi-resolution tiers: allocation policy,
coverage<->tokens inversion, and channel-major gather/scatter dispatch.

The fixed token budget per channel is spent non-uniformly by age: a geometric
patch-size ladder (PATCH = {4,8,16,64,256,512}) anchored at the forecast origin.
Tier ``k`` aggregates ``PATCH[k]`` consecutive raw steps per token; coverage
grows while tokens grow only linearly.

Allocation (D12): a *tunable ratio prior* (front-loaded, coarse-reaching;
config ``model.tier_alloc_per_channel``, normalized and scaled by n — default
reproduces the design point ``[8,8,8,8,8,4]`` at n=44, literal-halving available
as an ablation) then a deterministic integer rebalance so coverage ≈ T_raw —
moving single tokens between the finest and coarsest tiers (sum preserved).
Short series collapse toward patch-4; long series push tokens coarser.

Dispatch (D8/D14): per-tier raw-window slices + **channel-major** scatter
positions ``pos = channel_base + local_token_index`` (channel_base = base +
c*n_per_channel). Within a channel, tokens are laid out most-recent-first
(tier 0 first, then tier 1, ...). These index tensors feed the grouped per-tier
encode/scatter in the model (D14); pack time is CPU/numpy, no tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .constants import PATCH, V

# Allocation *ratio prior* over the V tiers (D12, clarified): per-tier counts are
# dynamic = normalized(prior) * n, rebalanced to T_raw. The default reproduces the
# design point — [16,16,16,16,16,8] normalizes so n=44 → [8,8,8,8,8,4]. Callers
# (the window sampler) pass cfg.model.tier_alloc_per_channel; this is the fallback.
DEFAULT_PRIOR: np.ndarray = np.array([16, 16, 16, 16, 16, 8], dtype=np.float64)
assert len(DEFAULT_PRIOR) == V


def _normalize_prior(prior=None) -> np.ndarray:
    p = DEFAULT_PRIOR if prior is None else np.asarray(prior, dtype=np.float64)
    if p.shape != (V,) or (p <= 0).any():
        raise ValueError(f"prior must be {V} positive weights; got {prior}")
    return p / p.sum()


def _largest_remainder(weights: np.ndarray, total: int) -> List[int]:
    """Integer apportionment summing exactly to ``total`` (largest-remainder)."""
    if total <= 0:
        return [0] * len(weights)
    raw = weights * total
    floor = np.floor(raw).astype(int)
    remainder = total - int(floor.sum())
    if remainder > 0:
        # hand the leftover units to the largest fractional parts
        order = np.argsort(-(raw - floor))
        for i in range(remainder):
            floor[order[i]] += 1
    return floor.tolist()


def default_counts(n_tokens: int, prior=None) -> List[int]:
    """Ratio-prior split of ``n_tokens`` across the V tiers (sums to n_tokens)."""
    return _largest_remainder(_normalize_prior(prior), int(n_tokens))


def coverage(counts: Sequence[int]) -> int:
    """Raw steps covered by a per-tier token allocation: Σ counts[k]·PATCH[k]."""
    return int(sum(int(c) * PATCH[k] for k, c in enumerate(counts)))


def allocate(n_tokens: int, t_raw: int, prior=None) -> List[int]:
    """Allocate ``n_tokens`` across tiers so coverage ≈ ``t_raw`` (D12).

    Starts from the ratio prior, then rebalances by moving single tokens between
    the finest (tier 0) and coarsest (tier V-1) tiers — total tokens preserved
    (≤ n_tokens budget; exactly n_tokens here). Deterministic.
    """
    counts = default_counts(n_tokens, prior)
    if n_tokens <= 0:
        return counts

    cov = coverage(counts)
    fine, coarse = 0, V - 1

    # Coverage too large (series shorter than tokens cover): shift coarse->fine,
    # reducing coverage. Short series become mostly patch-4.
    guard = 0
    while cov > t_raw and guard < 10 * n_tokens:
        guard += 1
        # take a token from the coarsest non-empty tier above the finest
        src = next((k for k in range(V - 1, 0, -1) if counts[k] > 0), None)
        if src is None:
            break
        nxt = counts.copy()
        nxt[src] -= 1
        nxt[fine] += 1
        ncov = coverage(nxt)
        # accept the move; if it overshoots below t_raw, keep whichever is closer
        if abs(ncov - t_raw) <= abs(cov - t_raw):
            counts, cov = nxt, ncov
        else:
            break

    # Coverage too small (series longer than coverage): shift fine->coarse,
    # increasing coverage, until coverage ≈ t_raw or the coarsest tier holds all.
    guard = 0
    while cov < t_raw and guard < 10 * n_tokens:
        guard += 1
        src = next((k for k in range(0, V - 1) if counts[k] > 0), None)
        if src is None:
            break
        nxt = counts.copy()
        nxt[src] -= 1
        nxt[coarse] += 1
        ncov = coverage(nxt)
        if abs(ncov - t_raw) <= abs(cov - t_raw):
            counts, cov = nxt, ncov
        else:
            break

    return counts


def tokens_for_coverage(t_raw: int, max_tokens: Optional[int] = None, prior=None) -> int:
    """Smallest token budget ``n`` whose ratio-prior coverage reaches ``t_raw``.

    Inverts the ladder for the D9 window sampler (``n = ⌊(L-Q)/C⌋`` fixes the
    budget; this returns the raw coverage achievable, used the other direction).
    Capped at ``max_tokens`` if given.
    """
    if t_raw <= 0:
        return 0
    n = 1
    while coverage(default_counts(n, prior)) < t_raw:
        n += 1
        if max_tokens is not None and n >= max_tokens:
            return max_tokens
    return n


@dataclass
class TierSlice:
    """Dispatch for one tier within one channel."""

    tier: int
    patch: int
    raw_start: np.ndarray   # [count] start raw index of each patch (relative to origin; may be < 0)
    raw_end: np.ndarray     # [count] end raw index (exclusive)
    scatter_pos: np.ndarray # [count] channel-major buffer positions


@dataclass
class ChannelDispatch:
    """Per-channel telescope dispatch (channel-major)."""

    counts: List[int]
    channel_base: int
    n_per_channel: int
    tiers: List[TierSlice]

    @property
    def positions(self) -> np.ndarray:
        """All scatter positions for this channel (token-order, most-recent-first)."""
        if not self.tiers:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([t.scatter_pos for t in self.tiers])


def build_dispatch(
    counts: Sequence[int],
    origin: int,
    channel_base: int,
    n_per_channel: int,
    caps: Optional[Sequence[int]] = None,
) -> ChannelDispatch:
    """Build channel-major gather/scatter dispatch for one channel.

    ``origin``: raw index one past the last context step (forecast origin).
    Tier ``k`` token ``i`` covers raw ``[origin - off_k - (i+1)P_k, origin - off_k - i*P_k)``
    where ``off_k = Σ_{j<k} counts[j]·PATCH[j]``. Patches that run off the start
    of available history (raw_start < 0) are emitted and masked by the caller.

    Scatter positions are contiguous within the channel block
    ``[channel_base, channel_base + Σcounts)`` ⊆ ``[channel_base, channel_base + n_per_channel)``.

    ``caps`` is an optional *convenience* guard (asserts ``counts[k] <= caps[k]``).
    The real static-capacity enforcement is the per-tier encoder dispatch capacity
    ``ENCODER_CAP`` (= L by default), applied in the model's Stage-7 routing: each
    encoder runs ``[CAP, P_k, 2]→[CAP, D]`` with sentinel padding, so any single
    tier can hold up to the whole context budget (e.g. a univariate buffer that is
    mostly one tier) with no dropped tokens.
    """
    counts = [int(c) for c in counts]
    if caps is not None:
        for k, (c, cap) in enumerate(zip(counts, caps)):
            if c > cap:
                raise ValueError(f"tier {k}: count {c} exceeds cap_k {cap}")
    total = sum(counts)
    if total > n_per_channel:
        raise ValueError(f"Σcounts {total} exceeds per-channel slot budget {n_per_channel}")

    tiers: List[TierSlice] = []
    local = 0       # running channel-major token index
    off = 0         # raw steps already covered by finer tiers
    for k, c in enumerate(counts):
        p = PATCH[k]
        if c == 0:
            tiers.append(
                TierSlice(k, p, np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
            )
            continue
        i = np.arange(c, dtype=np.int64)
        raw_end = origin - off - i * p
        raw_start = raw_end - p
        scatter = channel_base + local + i
        tiers.append(TierSlice(k, p, raw_start, raw_end, scatter))
        local += c
        off += c * p

    return ChannelDispatch(counts=counts, channel_base=channel_base,
                           n_per_channel=n_per_channel, tiers=tiers)


__all__ = [
    "PATCH",
    "V",
    "default_counts",
    "coverage",
    "allocate",
    "tokens_for_coverage",
    "TierSlice",
    "ChannelDispatch",
    "build_dispatch",
]
