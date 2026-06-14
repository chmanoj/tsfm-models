"""S4 test_tokenize — window sampler + assemble (walkthrough Stages 1–3).

Covers: spec-predicted S equals assembled length; origin/p/counts within bounds;
3-enum tagging (role CTX/QRY, content_state OBSERVED/MASK/NA, role_ft FEATURE/
TARGET); KFF = observed CTX feature token at t>0; flat norm store + raw_start
gather correctness; dense horizon targets; stats shapes; the mandated capacity
test (long univariate + 21-channel under one static ENCODER_CAP).
"""

import numpy as np
import torch

from tetris.constants import ChannelRole, ContentState, PATCH, Role, RoleFT, V
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window
import tetris.telescope as TS

BASE_PRIOR = (16, 16, 16, 16, 16, 8)
P_OUT = 16


def _params(l_pack=1024, p_out=P_OUT, **kw):
    return SamplerParams(l_pack=l_pack, p_out=p_out, tier_prior=BASE_PRIOR, **kw)


def _uni(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.cumsum(torch.randn(T, generator=g), dim=0).to(torch.float32)[None, :], 0, 1)


def _multi(C, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(C, T, generator=g).cumsum(dim=1).to(torch.float32), 0, C)


def _routed(seg):
    """Encoder-routed tokens = content_state OBSERVED (context + KFF)."""
    return seg.content_state == int(ContentState.OBSERVED)


def test_spec_S_equals_assembled_length():
    params = _params()
    for seed in range(25):
        rng = np.random.default_rng(seed)
        C = int(rng.integers(1, 12))
        T = int(rng.integers(64, 6000))
        item = _multi(C, T, seed)
        spec = sample_window(0, C, T, params, rng)
        seg = assemble(item, spec, P_OUT)
        assert seg.S == spec.S
        assert (seg.channel >= 0).all()                       # every slot assigned
        # observed tokens carry a raw_start; MASK/NA do not
        assert (seg.raw_start[_routed(seg)] >= 0).all()
        not_obs = seg.content_state != int(ContentState.OBSERVED)
        assert (seg.raw_start[not_obs] == -1).all()


def test_origin_p_counts_within_bounds():
    params = _params()
    for seed in range(25):
        rng = np.random.default_rng(seed + 100)
        C = int(rng.integers(1, 21))
        T = int(rng.integers(128, 8000))
        spec = sample_window(0, C, T, params, rng)
        assert 1 <= spec.origin <= T - spec.p
        assert 1 <= spec.p <= params.q_tok_max * params.p_out
        assert len(spec.counts) == V and 0 < spec.n_eff <= spec.n
        assert spec.Q_total == spec.Q + spec.K
        assert spec.n == (params.l_pack - spec.Q_total) // C
        assert len(spec.channel_roles) == C
        assert TS.coverage(spec.counts) <= spec.origin + PATCH[-1]


def test_raw_start_gather_in_bounds():
    # Each routed token's P_k window must lie inside the segment's norm store.
    params = _params()
    rng = np.random.default_rng(5)
    item = _multi(6, 4000, seed=9)
    spec = sample_window(0, 6, 4000, params, rng)
    seg = assemble(item, spec, P_OUT)
    R = seg.norm_values.shape[0]
    assert seg.observed.shape == (R,)
    for pos in np.nonzero(_routed(seg))[0]:
        Pk = PATCH[int(seg.tier_id[pos])]
        rs = int(seg.raw_start[pos])
        assert 0 <= rs and rs + Pk <= R


def test_enum_tagging_and_horizon_targets():
    # No NaN, all-target → context CTX/OBSERVED/TARGET, horizon QRY/MASK/TARGET.
    params = _params()
    rng = np.random.default_rng(7)
    item = _multi(4, 3000, seed=11)
    spec = sample_window(0, 4, 3000, params, rng)
    seg = assemble(item, spec, P_OUT)
    is_q = seg.role == int(Role.QRY)
    assert is_q.sum() == spec.Q
    assert (seg.content_state[is_q] == int(ContentState.MASK)).all()
    assert (seg.role_ft[is_q] == int(RoleFT.TARGET)).all()
    assert (seg.tier_id[is_q] == PATCH.index(P_OUT)).all()
    assert (seg.t_center[is_q] >= 0).all()
    # context tokens: CTX/OBSERVED (no missingness injected), t<0
    ctx = ~is_q
    assert (seg.role[ctx] == int(Role.CTX)).all()
    assert (seg.content_state[ctx] == int(ContentState.OBSERVED)).all()
    assert (seg.t_center[ctx] < 0).all()
    # dense horizon GT real only at query slots
    assert seg.horizon_target.shape == (seg.S, P_OUT)
    assert not seg.target_valid[~is_q].any()
    assert seg.target_valid[is_q].any()


def test_kff_is_observed_ctx_feature_at_future_time():
    # A feature channel revealed known-future → KFF tokens: role CTX, content
    # OBSERVED, role_ft FEATURE, t>0, routed to the P_out encoder. No KFF enum.
    params = _params(kff_reveal_prob=1.0)
    rng = np.random.default_rng(2)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 1200, generator=g).cumsum(dim=1).to(torch.float32)
    spec = sample_window(1, 1, 1200, params, rng)   # 1 feature, 1 target
    assert spec.channel_roles[0] == int(ChannelRole.KFF)
    assert spec.n_kff == 1 and spec.K == spec.q_tok
    seg = assemble((x, 1, 1), spec, P_OUT)
    # KFF tokens = future-time, CTX, OBSERVED, FEATURE, P_out tier
    kff = (seg.channel == 0) & (seg.t_center > 0)
    assert kff.sum() == spec.K
    assert (seg.role[kff] == int(Role.CTX)).all()
    assert (seg.content_state[kff] == int(ContentState.OBSERVED)).all()
    assert (seg.role_ft[kff] == int(RoleFT.FEATURE)).all()
    assert (seg.tier_id[kff] == PATCH.index(P_OUT)).all()
    assert (seg.raw_start[kff] >= 0).all()
    # KFF never produces query/horizon-target rows
    assert not seg.target_valid[kff].any()


def test_fully_missing_patch_becomes_na():
    params = _params()
    rng = np.random.default_rng(3)
    x = torch.randn(1, 2000).cumsum(dim=1).to(torch.float32)
    x[0, 1900:1990] = float("nan")  # blank a recent window fully
    spec = sample_window(0, 1, 2000, params, rng)
    seg = assemble((x, 0, 1), spec, P_OUT)
    ctx_lo = max(0, spec.origin - TS.coverage(spec.counts))
    blanked = np.isnan(x.numpy()[0, ctx_lo:spec.origin]).any()
    assert (int(ContentState.NA) in set(seg.content_state.tolist())) or (not blanked)


def test_stats_shapes():
    params = _params()
    rng = np.random.default_rng(8)
    spec = sample_window(0, 5, 2000, params, rng)
    seg = assemble(_multi(5, 2000, seed=4), spec, P_OUT)
    assert seg.stats_a.shape == (5,)
    assert seg.stats_sigma_delta.shape == (5, V)
    assert np.isfinite(seg.stats_a).all() and (seg.stats_sigma_delta > 0).all()


def test_capacity_univariate_and_21ch_same_encoder_cap():
    L_pack = 1024
    cap = L_pack  # ENCODER_CAP default (0 -> L_pack)
    params = _params(l_pack=L_pack, p_out=P_OUT)
    for item in [_uni(50000, seed=1), _multi(21, 8000, seed=2)]:
        C, T = item[0].shape
        rng = np.random.default_rng(C)
        spec = sample_window(item[1], item[2], T, params, rng)
        seg = assemble(item, spec, P_OUT)
        assert seg.S <= L_pack
        routed = _routed(seg)
        assert routed.sum() <= cap
        for k in range(V):
            assert int((seg.tier_id[routed] == k).sum()) <= cap
    rng = np.random.default_rng(2)
    spec21 = sample_window(0, 21, 8000, params, rng)
    assert 21 * spec21.n_eff <= cap
