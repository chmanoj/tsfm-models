"""S8 test_losses — horizon MAE masking, full-loss finiteness/differentiability,
and the record-only test-loss metric (D6/D13)."""

import os

import numpy as np
import torch

from tetris.config import load_config
from tetris.losses import compute_loss, horizon_loss
from tetris.metrics import horizon_test_loss
from tetris.model.tetris import Tetris
from tetris.packing.collator import pack
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
P_OUT, L_PACK = 8, 256
BASE_PRIOR = (16, 16, 16, 16, 16, 8)


def _cfg():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.packing.L_pack = L_PACK
    cfg.model.out_patch = P_OUT
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.model.n_heads = 2
    return cfg


def _seg(C, T, seed):
    g = torch.Generator().manual_seed(seed)
    item = (torch.randn(C, T, generator=g).cumsum(dim=1).to(torch.float32), 0, C)
    rng = np.random.default_rng(seed)
    spec = sample_window(0, C, T, SamplerParams(l_pack=L_PACK, p_out=P_OUT, tier_prior=BASE_PRIOR), rng)
    return assemble(item, spec, P_OUT)


def _batch():
    return pack([[_seg(2, 400, 1), _seg(3, 700, 2)], [_seg(1, 1800, 3)]], l_pack=L_PACK, p_out=P_OUT)


def test_full_loss_finite_and_differentiable():
    torch.manual_seed(0)
    cfg = _cfg()
    model = Tetris(cfg)
    b = _batch()
    out = model(b)
    lb = compute_loss(out, b, aux_weights=cfg.loss.aux_weights, loss_space=cfg.norm.loss_space)
    for s in (lb.total, lb.horizon, lb.aux_total):
        assert torch.isfinite(s) and s.ndim == 0
    assert lb.total >= 0
    lb.total.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )


def test_horizon_ignores_invalid_positions():
    b = _batch()
    pred = torch.randn(b.B, L_PACK, P_OUT)
    base = horizon_loss(pred, b)
    # Perturbing predictions only where target_valid is False must not change the loss.
    pred2 = pred.clone()
    pred2[~b.target_valid] += 123.0
    torch.testing.assert_close(horizon_loss(pred2, b), base)
    # ...but perturbing a valid position does change it.
    pred3 = pred.clone()
    vidx = b.target_valid.nonzero(as_tuple=False)[0]
    pred3[vidx[0], vidx[1], vidx[2]] += 5.0
    assert not torch.isclose(horizon_loss(pred3, b), base)


def test_metric_returns_float():
    torch.manual_seed(0)
    model = Tetris(_cfg())
    b = _batch()
    out = model(b)
    val = horizon_test_loss(out, b)
    assert isinstance(val, float) and np.isfinite(val)
