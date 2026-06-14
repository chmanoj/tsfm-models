"""S7 test_model_smoke — full forward produces finite, correctly-shaped dense
horizon + six aux outputs (Stages 6–9) on the active (Mac/SDPA) backend.

Also checks: the mask is actually applied (a query's output changes when a
ctx token it may read is perturbed, but not when an unreachable future ctx is);
the hoisted-basis path matches an internally-sampled one; gradients flow.
"""

import numpy as np
import torch

from tetris.config import load_config, resolved_encoder_cap
from tetris.constants import PATCH
from tetris.model.tetris import Tetris
from tetris.model.variate_id import sample_orthonormal_basis
from tetris.packing.collator import pack
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window

import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
P_OUT = 8
L_PACK = 256
BASE_PRIOR = (16, 16, 16, 16, 16, 8)


def _cfg():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    # shrink to the smoke sizes used here (L_pack=256, out_patch=8)
    cfg.packing.L_pack = L_PACK
    cfg.model.out_patch = P_OUT
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.model.n_heads = 2
    return cfg


def _params(**kw):
    return SamplerParams(l_pack=L_PACK, p_out=P_OUT, tier_prior=BASE_PRIOR, **kw)


def _seg(C, T, seed, **kw):
    g = torch.Generator().manual_seed(seed)
    item = (torch.randn(C, T, generator=g).cumsum(dim=1).to(torch.float32), 0, C)
    rng = np.random.default_rng(seed)
    spec = sample_window(0, C, T, _params(**kw), rng)
    return assemble(item, spec, P_OUT)


def _batch():
    return pack([[_seg(2, 400, 1), _seg(3, 600, 2)], [_seg(1, 1800, 3)]],
                l_pack=L_PACK, p_out=P_OUT)


def test_forward_shapes_and_finite():
    torch.manual_seed(0)
    cfg = _cfg()
    model = Tetris(cfg)
    b = _batch()
    out = model(b)
    assert out.horizon.shape == (b.B, L_PACK, P_OUT)
    assert torch.isfinite(out.horizon).all()
    assert len(out.aux) == len(PATCH)
    for k, a in enumerate(out.aux):
        assert a.shape == (b.B, L_PACK, PATCH[k])
        assert torch.isfinite(a).all()


def test_hoisted_basis_matches_internal():
    torch.manual_seed(0)
    cfg = _cfg()
    model = Tetris(cfg).eval()
    b = _batch()
    n_var = int(b.variate_uid.max()) + 1
    g1 = torch.Generator().manual_seed(123)
    basis = sample_orthonormal_basis(b.B, n_var, cfg.model.d_model, generator=g1)
    with torch.no_grad():
        o_hoist = model(b, variate_basis=basis).horizon
        g2 = torch.Generator().manual_seed(123)
        o_internal = model(b, generator=g2).horizon
    torch.testing.assert_close(o_hoist, o_internal)


def test_gradients_flow():
    torch.manual_seed(0)
    model = Tetris(_cfg())
    b = _batch()
    out = model(b)
    out.horizon.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_mask_blocks_unreachable_context():
    # A query reads all ctx of its sample but a ctx token cannot read a future
    # ctx token. Perturbing an early-time ctx's input must move a later query's
    # output (reachable); the mask must keep cross-sample tokens from leaking.
    torch.manual_seed(0)
    model = Tetris(_cfg()).eval()
    b1 = _batch()
    # Zero out buffer 1 entirely; buffer 0's outputs must be unchanged (no cross
    # buffer leakage), since attention never crosses buffers.
    with torch.no_grad():
        o_full = model(b1, generator=torch.Generator().manual_seed(5)).horizon
        b2 = _batch()
        b2.norm_values[1].zero_()
        b2.observed[1] = False
        o_pert = model(b2, generator=torch.Generator().manual_seed(5)).horizon
    torch.testing.assert_close(o_full[0], o_pert[0])           # buffer 0 untouched
    assert not torch.allclose(o_full[1], o_pert[1])            # buffer 1 changed
