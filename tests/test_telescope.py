"""[GATE] test_telescope — D2/D12 allocation + channel-major dispatch.

Covers: allocation within budget & coverage≈T_raw; short series collapse to
patch-4, long series push coarse; channel-major positions collision-free;
scatter(gather) round-trips; cap_k asserts.
"""

import numpy as np
import pytest

from tetris import telescope as TS
from tetris.constants import PATCH, V


def test_default_counts_sum_exactly():
    for n in [0, 1, 5, 44, 100, 333]:
        c = TS.default_counts(n)
        assert len(c) == V
        assert sum(c) == n
        assert all(x >= 0 for x in c)


def test_allocate_budget_and_coverage_close():
    n = 44
    for t_raw in [50, 200, 1232, 5000, 20000]:
        counts = TS.allocate(n, t_raw)
        assert sum(counts) == n               # budget fully + exactly used
        assert all(x >= 0 for x in counts)
        cov = TS.coverage(counts)
        at_floor = counts[0] == n             # collapsed to patch-4 (short series)
        at_ceil = counts[-1] == n             # piled into coarsest (long series)
        if not (at_floor or at_ceil):
            # rebalance moves single tokens between finest/coarsest tiers, so the
            # residual is bounded by one coarse-patch step (D12 "coverage ≈ T_raw").
            assert abs(cov - t_raw) <= PATCH[-1]


def test_short_series_collapse_to_patch4():
    # T_raw far below the minimum coverage achievable -> all tokens at patch-4.
    counts = TS.allocate(44, t_raw=10)
    assert counts[0] == 44
    assert sum(counts[1:]) == 0


def test_long_series_push_coarse():
    # T_raw near the maximum coverage -> tokens pile into the coarsest tier.
    counts = TS.allocate(44, t_raw=10_000_000)
    assert counts[-1] == 44
    assert sum(counts[:-1]) == 0
    assert TS.coverage(counts) == 44 * PATCH[-1]


def test_tokens_for_coverage_monotone_and_sufficient():
    for t in [1, 100, 1232, 4832, 50000]:
        n = TS.tokens_for_coverage(t)
        assert TS.coverage(TS.default_counts(n)) >= t
        if n > 0:
            assert TS.coverage(TS.default_counts(n - 1)) < t


def test_dispatch_positions_collision_free_and_in_block():
    n_per_channel = 50
    counts = TS.allocate(44, t_raw=1232)
    C = 5
    all_pos = []
    for c in range(C):
        base = c * n_per_channel
        disp = TS.build_dispatch(counts, origin=2000, channel_base=base,
                                 n_per_channel=n_per_channel, caps=None)
        pos = disp.positions
        # within this channel's block, contiguous from base
        assert pos.min() >= base
        assert pos.max() < base + n_per_channel
        assert pos.max() < base + sum(counts)
        all_pos.append(pos)
    cat = np.concatenate(all_pos)
    assert len(np.unique(cat)) == cat.size          # globally collision-free
    assert cat.size == C * sum(counts)


def test_scatter_gather_roundtrip():
    # A token-feature buffer scattered to dispatch positions and gathered back
    # must recover the originals exactly (positions form a clean partition).
    n_per_channel = 50
    counts = TS.allocate(44, t_raw=1232)
    C = 3
    L = C * n_per_channel
    buf = np.full(L, -1, dtype=np.int64)
    payloads = {}
    for c in range(C):
        base = c * n_per_channel
        disp = TS.build_dispatch(counts, origin=2000, channel_base=base, n_per_channel=n_per_channel)
        pos = disp.positions
        vals = np.arange(pos.size, dtype=np.int64) + c * 1000
        buf[pos] = vals
        payloads[c] = (pos, vals)
    for c, (pos, vals) in payloads.items():
        gathered = buf[pos]
        assert np.array_equal(gathered, vals)


def test_dispatch_raw_windows_contiguous_and_aligned():
    # Patches abut without gaps/overlap and tile backward from the origin.
    counts = [8, 8, 8, 8, 8, 4]  # D12 design point
    origin = 100000
    disp = TS.build_dispatch(counts, origin=origin, channel_base=0, n_per_channel=64)
    edges = []
    for ts in disp.tiers:
        for s, e in zip(ts.raw_start.tolist(), ts.raw_end.tolist()):
            assert e - s == ts.patch
            edges.append((s, e))
    # most-recent patch ends at the origin
    assert edges[0][1] == origin
    # contiguous tiling: each patch starts where the previous (more recent) ended
    for (s0, e0), (s1, e1) in zip(edges[:-1], edges[1:]):
        assert e1 == s0
    # total span == coverage
    assert edges[0][1] - edges[-1][0] == TS.coverage(counts)


def test_cap_assert_raises():
    counts = [20, 0, 0, 0, 0, 0]
    caps = [16, 12, 8, 6, 4, 2]  # tier-0 count 20 > cap 16
    with pytest.raises(ValueError):
        TS.build_dispatch(counts, origin=1000, channel_base=0, n_per_channel=64, caps=caps)


def test_slot_overflow_raises():
    counts = TS.allocate(44, t_raw=1232)
    with pytest.raises(ValueError):
        TS.build_dispatch(counts, origin=1000, channel_base=0, n_per_channel=10)
