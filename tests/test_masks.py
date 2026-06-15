"""[GATE] test_masks — D9.1 attention truth table; Flex == SDPA.

Enumerates a small synthetic grid and asserts every (q, k) pair matches the D9
boolean formula, covering each rule category explicitly: ctx→ctx causal +
channel-blind, qry→ctx always, qry→qry bidirectional, ctx→qry never, KFF-as-ctx
(observed CTX at t>0), cross-sample/cross-buffer isolation, and pad self-only.
Then asserts the SDPA bool mask equals the enumerated Flex ``mask_mod``, and that
the Flex ``BlockMask`` constructor builds.
"""

import torch

from tetris.constants import PAD_SAMPLE_ID, Role
from tetris.masks import (
    build_block_mask,
    build_sdpa_mask,
    dense_from_mask_mod,
    make_mask_mod,
)

CTX = int(Role.CTX)
QRY = int(Role.QRY)


def _grid():
    """A 2-buffer grid (L=6). Buffer 0: one sample with cross-channel ctx, a KFF
    ctx token at t=1, and two queries. Buffer 1: two packed samples + pad tail."""
    # buffer 0  (single sample 0)
    #   idx: 0      1      2      3(KFF) 4      5
    s0 = [0, 0, 0, 0, 0, 0]
    r0 = [CTX, CTX, CTX, CTX, QRY, QRY]
    t0 = [-3.0, -2.0, -2.0, 1.0, 0.0, 1.0]
    # buffer 1  (samples 1 and 2, then two pad slots)
    s1 = [1, 1, 2, 2, PAD_SAMPLE_ID, PAD_SAMPLE_ID]
    r1 = [CTX, QRY, CTX, QRY, CTX, CTX]  # pad role is arbitrary (masked by sample_id)
    t1 = [-1.0, 0.0, -1.0, 0.0, 0.0, 0.0]

    sample_id = torch.tensor([s0, s1], dtype=torch.int32)
    role = torch.tensor([r0, r1], dtype=torch.int8)
    t_center = torch.tensor([t0, t1], dtype=torch.float32)
    return sample_id, role, t_center


def _ref_allow(s, r, t, b, q, k):
    """Reference D9 formula, scalar, including the pad self-attend exception."""
    if s[b][q] == PAD_SAMPLE_ID:
        return q == k
    same = s[b][q] == s[b][k]
    ctx_rule = (r[b][k] == CTX) and ((r[b][q] == QRY) or (t[b][k] <= t[b][q]))
    qry_rule = (r[b][k] == QRY) and (r[b][q] == QRY)
    return same and (ctx_rule or qry_rule)


def test_sdpa_matches_formula_by_enumeration():
    sample_id, role, t_center = _grid()
    B, L = sample_id.shape
    mask = build_sdpa_mask(sample_id, role, t_center)
    assert mask.shape == (B, 1, L, L)
    assert mask.dtype == torch.bool

    s, r, t = sample_id.tolist(), role.tolist(), t_center.tolist()
    for b in range(B):
        for q in range(L):
            for k in range(L):
                assert bool(mask[b, 0, q, k]) == _ref_allow(s, r, t, b, q, k), (
                    f"mismatch at b={b} q={q} k={k}"
                )


def test_flex_equals_sdpa():
    sample_id, role, t_center = _grid()
    B, L = sample_id.shape
    sdpa = build_sdpa_mask(sample_id, role, t_center)[:, 0]  # [B, L, L]
    mod = make_mask_mod(sample_id, role, t_center)
    flex_dense = dense_from_mask_mod(mod, B, L)
    assert torch.equal(sdpa, flex_dense)


def test_block_mask_constructor_builds():
    # On CPU the BlockMask is block-granular; we only assert it constructs (the
    # exact rule is checked elementwise above). On CUDA this is the live kernel.
    sample_id, role, t_center = _grid()
    bm = build_block_mask(sample_id, role, t_center)
    assert bm is not None


def test_ctx_to_ctx_causal_and_channel_blind():
    sample_id, role, t_center = _grid()
    m = build_sdpa_mask(sample_id, role, t_center)[:, 0]
    # buffer 0: idx1 (ctx, t=-2) attends idx0 (ctx, t=-3): causal past -> True
    assert m[0, 1, 0]
    # ...and itself (t<=t) -> True
    assert m[0, 1, 1]
    # idx1 (ctx, ch0, t=-2) attends idx2 (ctx, ch1, t=-2): channel-blind, t==t -> True
    assert m[0, 1, 2]
    # ctx cannot see the future: idx1 (t=-2) -> idx3 (KFF ctx, t=1) -> False
    assert not m[0, 1, 3]


def test_qry_to_ctx_always_and_kff():
    sample_id, role, t_center = _grid()
    m = build_sdpa_mask(sample_id, role, t_center)[:, 0]
    # qry idx4 (t=0) reads every ctx in its sample regardless of time, incl. KFF(t=1)
    for k in (0, 1, 2, 3):
        assert m[0, 4, k]
    # qry idx5 reads the KFF ctx too
    assert m[0, 5, 3]


def test_qry_to_qry_bidirectional_and_ctx_to_qry_never():
    sample_id, role, t_center = _grid()
    m = build_sdpa_mask(sample_id, role, t_center)[:, 0]
    # qry<->qry both directions (idx4, idx5)
    assert m[0, 4, 5]
    assert m[0, 5, 4]
    # ctx never attends a query: idx3 (KFF ctx) -> idx4/idx5 (qry) -> False
    assert not m[0, 3, 4]
    assert not m[0, 3, 5]


def test_cross_sample_and_pad_isolation():
    sample_id, role, t_center = _grid()
    m = build_sdpa_mask(sample_id, role, t_center)[:, 0]
    # buffer 1: sample 1 (idx0,1) cannot attend sample 2 (idx2,3) though packed together
    for q in (0, 1):
        for k in (2, 3):
            assert not m[1, q, k]
            assert not m[1, k, q]
    # pad rows (idx4, idx5) attend only themselves
    for p in (4, 5):
        for k in range(6):
            assert bool(m[1, p, k]) == (k == p)
    # no real token attends a pad key
    for q in range(4):
        for p in (4, 5):
            assert not m[1, q, p]
