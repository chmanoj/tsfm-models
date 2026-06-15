"""[GATE] test_pack_invariance — the D8 no-leakage keystone (S9).

Identical samples packed together vs in solo buffers (and in different orders /
groupings) must produce **identical per-sample token outputs and total loss**.
This is the guard against buffer-index leakage (D8: token geometry comes only from
side tensors) and confirms cross-resolution aux overlap (OA) is layout-independent.

Random variate IDs (D4) are *relational*, so a single forward is not invariant to
which orthonormal vectors are drawn — only relationally over resampling. For the
test we therefore pin a **fixed basis** that maps each sample's channels to the
same ID vectors in every layout (the model resamples freely in training); with
that pinned, true no-leakage ⇒ bit-identical per-sample outputs.
"""

import os

import numpy as np
import torch

from tetris.config import load_config
from tetris.losses import compute_loss
from tetris.model.tetris import Tetris
from tetris.packing.collator import pack
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
P_OUT, L_PACK, D = 8, 512, 32
PRIOR = (16, 16, 16, 16, 16, 8)
AUX_W = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

# three fixed samples (multivariate mix); global channel offsets for the ID pool
_SPECS = [(2, 2500, 11), (1, 4000, 12), (3, 1800, 13)]


def _seg(C, T, seed):
    g = torch.Generator().manual_seed(seed)
    item = (torch.randn(C, T, generator=g).cumsum(1).to(torch.float32), 0, C)
    rng = np.random.default_rng(seed)
    spec = sample_window(0, C, T, SamplerParams(l_pack=L_PACK, p_out=P_OUT, tier_prior=PRIOR), rng)
    return assemble(item, spec, P_OUT)


def _segments():
    return [_seg(*s) for s in _SPECS]


def _offsets(segs):
    off, acc = [], 0
    for s in segs:
        off.append(acc)
        acc += s.C
    return off, acc


def _build_basis(layout_segs, layout_gids, pool, offsets):
    B = len(layout_segs)
    n_var = max((sum(s.C for s in segs) for segs in layout_segs), default=1) or 1
    basis = torch.zeros(B, n_var, D)
    for bi, (segs, gids) in enumerate(zip(layout_segs, layout_gids)):
        uid = 0
        for seg, gid in zip(segs, gids):
            for c in range(seg.C):
                basis[bi, uid] = pool[offsets[gid] + c]
                uid += 1
    return basis


def _run(model, layout_segs, layout_gids, pool, offsets):
    batch = pack(layout_segs, l_pack=L_PACK, p_out=P_OUT)
    basis = _build_basis(layout_segs, layout_gids, pool, offsets)
    with torch.no_grad():
        out = model(batch, variate_basis=basis)
    per_sample = {}
    for bi, (segs, gids) in enumerate(zip(layout_segs, layout_gids)):
        base = 0
        for seg, gid in zip(segs, gids):
            sl = slice(base, base + seg.S)
            per_sample[gid] = (
                out.horizon[bi, sl].clone(),
                [a[bi, sl].clone() for a in out.aux],
            )
            base += seg.S
    return batch, out, per_sample


def _assert_sample_equal(a, b):
    torch.testing.assert_close(a[0], b[0], rtol=1e-4, atol=1e-5)
    for ka, kb in zip(a[1], b[1]):
        torch.testing.assert_close(ka, kb, rtol=1e-4, atol=1e-5)


def _model():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.packing.L_pack = L_PACK
    cfg.model.out_patch = P_OUT
    cfg.model.d_model = D
    cfg.model.n_layers = 2
    cfg.model.n_heads = 2
    torch.manual_seed(0)
    return Tetris(cfg).eval()


def test_solo_vs_packed_together_per_sample_identical():
    model = _model()
    segs = _segments()
    _, total_c = _offsets(segs)
    offsets, _ = _offsets(segs)
    pool = torch.randn(total_c, D, generator=torch.Generator().manual_seed(99))

    solo = ([[segs[0]], [segs[1]], [segs[2]]], [[0], [1], [2]])
    together = ([[segs[0], segs[1], segs[2]]], [[0, 1, 2]])

    _, out_s, ps_solo = _run(model, *solo, pool, offsets)
    bt, out_t, ps_tog = _run(model, *together, pool, offsets)

    for gid in (0, 1, 2):
        _assert_sample_equal(ps_solo[gid], ps_tog[gid])

    # sanity: the comparison is non-trivial (different samples differ)
    assert not torch.allclose(ps_solo[0][0], ps_solo[2][0][: ps_solo[0][0].shape[0]])


def test_permutation_and_regrouping_invariant():
    model = _model()
    segs = _segments()
    offsets, _ = _offsets(segs)
    pool = torch.randn(sum(s.C for s in segs), D, generator=torch.Generator().manual_seed(7))

    solo = ([[segs[0]], [segs[1]], [segs[2]]], [[0], [1], [2]])
    permuted = ([[segs[2], segs[0], segs[1]]], [[2, 0, 1]])          # different order
    regrouped = ([[segs[0], segs[2]], [segs[1]]], [[0, 2], [1]])     # different grouping

    _, _, ps_solo = _run(model, *solo, pool, offsets)
    _, _, ps_perm = _run(model, *permuted, pool, offsets)
    _, _, ps_reg = _run(model, *regrouped, pool, offsets)

    for gid in (0, 1, 2):
        _assert_sample_equal(ps_solo[gid], ps_perm[gid])
        _assert_sample_equal(ps_solo[gid], ps_reg[gid])


def test_total_loss_layout_invariant():
    model = _model()
    segs = _segments()
    offsets, total_c = _offsets(segs)
    pool = torch.randn(total_c, D, generator=torch.Generator().manual_seed(5))

    solo = ([[segs[0]], [segs[1]], [segs[2]]], [[0], [1], [2]])
    together = ([[segs[0], segs[1], segs[2]]], [[0, 1, 2]])

    batch_s, out_s, _ = _run(model, *solo, pool, offsets)
    batch_t, out_t, _ = _run(model, *together, pool, offsets)

    lb_s = compute_loss(out_s, batch_s, aux_weights=AUX_W)
    lb_t = compute_loss(out_t, batch_t, aux_weights=AUX_W)
    torch.testing.assert_close(lb_s.total, lb_t.total, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lb_s.horizon, lb_t.horizon, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lb_s.aux_total, lb_t.aux_total, rtol=1e-4, atol=1e-5)
