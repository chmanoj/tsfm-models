"""S4 test_tokenize — window sampler + assemble.

Covers: spec-predicted S equals assembled length exactly; origin/p/counts within
bounds; content_state assignment correct (query→MASK, observed→OBSERVED,
fully-missing→NA). Plus the mandated capacity test: a long univariate crop AND a
21-channel crop both dispatch correctly under the same static n_ctx_cap
(counts-vs-capacity split).
"""

import numpy as np
import torch

from tetris.constants import ContentState, PATCH, Role, RoleFT, V
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window
import tetris.telescope as TS

BASE_PRIOR = (16, 16, 16, 16, 16, 8)


def _params(l_pack=1024, p_out=16, **kw):
    return SamplerParams(l_pack=l_pack, p_out=p_out, tier_prior=BASE_PRIOR, **kw)


def _uni(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.cumsum(torch.randn(T, generator=g), dim=0).to(torch.float32)
    return (x[None, :], 0, 1)


def _multi(C, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(C, T, generator=g).cumsum(dim=1).to(torch.float32)
    return (x, 0, C)


def test_spec_S_equals_assembled_length():
    params = _params()
    for seed in range(25):
        rng = np.random.default_rng(seed)
        C = int(rng.integers(1, 12))
        T = int(rng.integers(64, 6000))
        item = _multi(C, T, seed)
        spec = sample_window(0, C, T, params, rng)
        seg = assemble(item, spec)
        assert seg.S == spec.S
        # every position is assigned exactly once (no holes, no -1)
        assert (seg.channel_idx >= 0).all()
        # encoder-routed tokens (context + KFF horizon) + queries == S
        routed = sum(int(p.size) for p in seg.tier_pos)
        assert routed + seg.qry_pos.size == seg.S


def test_origin_p_counts_within_bounds():
    params = _params()
    for seed in range(25):
        rng = np.random.default_rng(seed + 100)
        C = int(rng.integers(1, 21))
        T = int(rng.integers(128, 8000))
        spec = sample_window(0, C, T, params, rng)
        assert 1 <= spec.origin <= T - spec.p
        assert 1 <= spec.p <= params.q_tok_max * params.p_out
        assert len(spec.counts) == V and sum(spec.counts) == spec.n_eff
        assert all(c >= 0 for c in spec.counts)
        nc = spec.n_horizon_obs + spec.Q
        assert spec.n_eff <= (params.l_pack - nc) // C
        # coverage cannot exceed available history (no pre-start tokens)
        assert TS.coverage(spec.counts) <= spec.origin + PATCH[-1]


def test_content_state_assignment():
    # No role augmentation, no NaN → context OBSERVED, horizon all queries (MASK).
    params = _params(role_aug_prob=0.0)
    rng = np.random.default_rng(7)
    item = _multi(4, 3000, seed=11)
    spec = sample_window(0, 4, 3000, params, rng)
    seg = assemble(item, spec)
    is_qry = seg.role == int(Role.QRY)
    is_ctx = seg.role == int(Role.CTX)
    assert (seg.content_state[is_qry] == int(ContentState.MASK)).all()
    assert (seg.content_state[is_ctx] == int(ContentState.OBSERVED)).all()
    # queries only on target channels; all 4 channels are targets here
    assert (seg.role_ft[is_qry] == int(RoleFT.TARGET)).all()
    assert seg.qry_pos.size == spec.n_target_eff * spec.q_tok


def test_fully_missing_patch_becomes_na():
    params = _params(role_aug_prob=0.0)
    rng = np.random.default_rng(3)
    x = torch.randn(1, 2000).cumsum(dim=1).to(torch.float32)
    # blank a contiguous recent window fully -> at least one tier-0 patch all-NaN
    x[0, 1900:1980] = float("nan")
    spec = sample_window(0, 1, 2000, params, rng)
    seg = assemble((x, 0, 1), spec)
    # if any token's window fell entirely in the blanked region it must be NA
    assert int(ContentState.NA) in set(seg.content_state.tolist()) or not np.isnan(
        x.numpy()[0, max(0, spec.origin - TS.coverage(spec.counts)):spec.origin]
    ).any()


def test_capacity_univariate_and_21ch_same_n_ctx_cap():
    # counts-vs-capacity split: both crops dispatch under one static n_ctx_cap.
    L_pack = 1024
    n_ctx_cap = L_pack  # default resolution (0 -> L_pack)
    params = _params(l_pack=L_pack, p_out=16)

    cases = [_uni(50000, seed=1), _multi(21, 8000, seed=2)]
    for item in cases:
        C = item[0].shape[0]
        T = item[0].shape[1]
        rng = np.random.default_rng(C)
        spec = sample_window(item[1], item[2], T, params, rng)
        seg = assemble(item, spec)
        # total buffer occupancy fits L_pack
        assert seg.S <= L_pack
        # encoder-routed tokens (one per context/KFF slot) fit the static slot space
        routed = sum(int(p.size) for p in seg.tier_pos)
        assert routed <= n_ctx_cap
        # every tier's index tensor (length cap_k = n_ctx_cap) holds its tokens
        for k in range(V):
            assert seg.tier_pos[k].size <= n_ctx_cap
            assert seg.tier_values[k].shape == (seg.tier_pos[k].size, PATCH[k])
    # the 21-channel crop reproduces the D12 design point envelope
    rng = np.random.default_rng(2)
    spec21 = sample_window(0, 21, 8000, params, rng)
    assert 21 * spec21.n_eff <= n_ctx_cap
