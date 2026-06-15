"""S6 test_collator_shapes — stateless pack() materializes the Batch (Stage 4).

Covers: static [B,L] / [B,R] / [B,L,P_out] shapes; tail pad sample_id=-1 (and pad
defaults); per-buffer sample_id numbering; raw_start rebased into the buffer store
(gather round-trips the per-segment window); variate_uid buffer-unique per
(sample,channel); per-token stats broadcast; dense horizon GT placement; extra
buffers all-pad; overflow raises.
"""

import numpy as np
import pytest
import torch

from tetris.constants import ContentState, PAD_SAMPLE_ID, PATCH
from tetris.packing.collator import pack
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window

P_OUT = 16
L_PACK = 1024
BASE_PRIOR = (16, 16, 16, 16, 16, 8)


def _params(**kw):
    return SamplerParams(l_pack=L_PACK, p_out=P_OUT, tier_prior=BASE_PRIOR, **kw)


def _seg(C, T, seed):
    g = torch.Generator().manual_seed(seed)
    item = (torch.randn(C, T, generator=g).cumsum(dim=1).to(torch.float32), 0, C)
    rng = np.random.default_rng(seed)
    spec = sample_window(0, C, T, _params(), rng)
    return assemble(item, spec, P_OUT)


def test_shapes_and_pad_defaults():
    s0 = _seg(2, 400, 1)
    s1 = _seg(3, 600, 2)
    s2 = _seg(1, 2000, 3)
    buffers = [[s0, s1], [s2]]  # buf0 packs two segments, buf1 one
    b = pack(buffers, l_pack=L_PACK, p_out=P_OUT)

    assert b.B == 2 and b.L == L_PACK
    for t in (b.sample_id, b.channel_idx, b.t_center, b.tier_id, b.role,
              b.content_state, b.role_ft, b.raw_start, b.variate_uid, b.valid_aux,
              b.stats_a, b.stats_sigma):
        assert t.shape == (2, L_PACK)
    assert b.norm_values.shape == (2, b.R)
    assert b.observed.shape == (2, b.R)
    assert b.horizon_target.shape == (2, L_PACK, P_OUT)
    assert b.target_valid.shape == (2, L_PACK, P_OUT)

    # buffer 0: segment 0 occupies [0,S0), segment 1 [S0, S0+S1), then pad
    S0, S1 = s0.S, s1.S
    sid = b.sample_id[0].numpy()
    assert (sid[:S0] == 0).all()
    assert (sid[S0:S0 + S1] == 1).all()
    assert (sid[S0 + S1:] == PAD_SAMPLE_ID).all()
    # pad defaults: NA content, no raw window, no variate, invalid GT
    pad = sid == PAD_SAMPLE_ID
    assert (b.content_state[0].numpy()[pad] == int(ContentState.NA)).all()
    assert (b.raw_start[0].numpy()[pad] == -1).all()
    assert (b.variate_uid[0].numpy()[pad] == -1).all()
    assert not b.target_valid[0][torch.from_numpy(pad)].any()


def test_raw_start_rebased_gather_roundtrips():
    # The buffer-store gather at a token's rebased raw_start must equal the
    # per-segment store window at its original raw_start (offset == placement).
    s0 = _seg(2, 500, 11)
    s1 = _seg(2, 900, 12)
    b = pack([[s0, s1]], l_pack=L_PACK, p_out=P_OUT)
    nv = b.norm_values[0].numpy()

    S0 = s0.S
    store0 = s0.norm_values.shape[0]
    for local_id, (seg, tok_base, store_base) in enumerate(
        [(s0, 0, 0), (s1, S0, store0)]
    ):
        routed = np.nonzero(seg.content_state == int(ContentState.OBSERVED))[0]
        for pos in routed[:50]:
            Pk = PATCH[int(seg.tier_id[pos])]
            seg_rs = int(seg.raw_start[pos])
            buf_rs = int(b.raw_start[0, tok_base + pos])
            assert buf_rs == seg_rs + store_base
            assert 0 <= buf_rs and buf_rs + Pk <= b.R
            np.testing.assert_allclose(
                nv[buf_rs:buf_rs + Pk], seg.norm_values[seg_rs:seg_rs + Pk], rtol=0, atol=0
            )


def test_variate_uid_buffer_unique_per_sample_channel():
    s0 = _seg(3, 400, 21)
    s1 = _seg(2, 700, 22)
    b = pack([[s0, s1]], l_pack=L_PACK, p_out=P_OUT)
    sid = b.sample_id[0].numpy()
    ch = b.channel_idx[0].numpy()
    vu = b.variate_uid[0].numpy()
    real = sid != PAD_SAMPLE_ID

    # bijection: each (sample, channel) maps to exactly one uid and vice-versa
    pairs = {}
    for s, c, u in zip(sid[real], ch[real], vu[real]):
        pairs.setdefault((s, c), set()).add(u)
    assert all(len(us) == 1 for us in pairs.values())          # one uid per (s,c)
    uid_for_pair = {k: next(iter(v)) for k, v in pairs.items()}
    assert len(set(uid_for_pair.values())) == len(uid_for_pair)  # distinct across pairs
    # second segment's channels are offset past the first segment's C=3
    assert uid_for_pair[(1, 0)] == 3


def test_stats_broadcast_per_token():
    s0 = _seg(4, 1500, 31)
    b = pack([[s0]], l_pack=L_PACK, p_out=P_OUT)
    ch = b.channel_idx[0].numpy()
    real = ch >= 0
    np.testing.assert_allclose(b.stats_a[0].numpy()[real], s0.stats_a[ch[real]])
    np.testing.assert_allclose(
        b.stats_sigma[0].numpy()[real], s0.stats_sigma_delta[ch[real], 0]
    )
    assert (b.stats_sigma[0].numpy()[real] > 0).all()


def test_extra_buffers_all_pad():
    s0 = _seg(1, 800, 41)
    b = pack([[s0]], l_pack=L_PACK, p_out=P_OUT, num_buffers=3)
    assert b.B == 3
    assert (b.sample_id[1].numpy() == PAD_SAMPLE_ID).all()
    assert (b.sample_id[2].numpy() == PAD_SAMPLE_ID).all()
    assert (b.raw_start[1].numpy() == -1).all()


def test_too_many_buffers_and_overflow_raise():
    s_big = _seg(1, 80000, 51)  # deep univariate ≈ fills a buffer
    with pytest.raises(ValueError):
        pack([[s_big], [s_big]], l_pack=L_PACK, p_out=P_OUT, num_buffers=1)
    # two near-full segments in one buffer overflow L
    with pytest.raises(ValueError):
        pack([[s_big, s_big]], l_pack=L_PACK, p_out=P_OUT)
