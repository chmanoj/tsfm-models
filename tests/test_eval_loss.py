"""S12 test_eval_loss — record-only GIFT-Eval test loss (§6), offline.

Covers: the synthetic eval shard honors the EvalItem contract and strips to the
training Item; ``eval_batch`` packs context-only via the **unchanged** collator
with the held-out horizon landing only in ``horizon_target`` (query slots are
MASK — no leakage into the observed store); ``evaluate_test_loss`` is finite and
record-only; and the training loop runs the eval hook without breaking the
training-loader path. The real network download is intentionally not exercised.
"""

import os

import numpy as np
import torch

from tetris.config import load_config
from tetris.constants import ContentState, Role
from tetris.data.contract import build_eval_loader, to_train_item, validate_item
from tetris.data.eval_loader import (
    GiftEvalEvalLoader,
    eval_batch,
    evaluate_test_loss,
    make_synthetic_eval_shard,
)
from tetris.model.tetris import Tetris
from tetris.tokenize.window_sampler import SamplerParams
from tetris.train.loop import run_training

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
P_OUT = 8


def _cfg():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.packing.L_pack = 256
    cfg.packing.buffers_per_step = 2
    cfg.model.out_patch = P_OUT
    cfg.model.d_model = 16
    cfg.model.n_layers = 1
    cfg.model.n_heads = 2
    cfg.eval.loader = "synthetic_eval"
    cfg.eval.shard_windows = 6
    cfg.data.n_series = 200
    cfg.data.length_distribution = [64, 256]
    cfg.data.C_distribution = [1, 4]
    return cfg


def _params(cfg):
    return SamplerParams(l_pack=cfg.packing.L_pack, p_out=cfg.model.out_patch,
                         tier_prior=tuple(cfg.model.tier_alloc_per_channel))


def test_synthetic_shard_honors_contract():
    cfg = _cfg()
    shard = make_synthetic_eval_shard(cfg, n_items=6, seed=0)
    assert len(shard) == 6
    for e in shard:
        # First three fields are exactly the training Item.
        validate_item(to_train_item(e))
        assert e.y_true.ndim == 2 and e.y_true.shape[1] == e.num_targets
        assert e.naive_denom is None  # MASE deferred (O4)
        assert e.data_tensor.shape[0] == e.num_features + e.num_targets


def test_eval_batch_holds_out_horizon_no_leakage():
    cfg = _cfg()
    e = make_synthetic_eval_shard(cfg, n_items=1, seed=1)[0]
    batch = eval_batch(e, _params(cfg), p_out=P_OUT, l_pack=cfg.packing.L_pack)

    # The held-out horizon is scored: query slots exist and carry valid targets.
    qry = batch.role == int(Role.QRY)
    assert qry.any()
    assert batch.target_valid.any()
    # No leakage: every query slot is MASK content (y_true only in horizon_target,
    # never the observed store), and no OBSERVED token sits at/after the origin.
    assert bool((batch.content_state[qry] == int(ContentState.MASK)).all())
    observed = batch.content_state == int(ContentState.OBSERVED)
    assert bool((batch.t_center[observed] < 0).all())


def test_evaluate_test_loss_is_finite_record_only():
    cfg = _cfg()
    torch.manual_seed(0)
    model = Tetris(cfg)
    loader = GiftEvalEvalLoader.from_synthetic(cfg, n_items=cfg.eval.shard_windows)
    before = [p.detach().clone() for p in model.parameters()]
    tl = evaluate_test_loss(model, loader, cfg)
    assert np.isfinite(tl) and tl >= 0.0
    # Record-only: parameters are untouched.
    for p0, p1 in zip(before, model.parameters()):
        assert torch.equal(p0, p1)


def test_build_eval_loader_synthetic_key():
    cfg = _cfg()
    loader = build_eval_loader(cfg)
    assert isinstance(loader, GiftEvalEvalLoader)
    assert len(loader) == cfg.eval.shard_windows


def test_training_loop_eval_hook_unbroken():
    cfg = _cfg()
    cfg.data.loader = "standin_pretrain"
    cfg.packing.reservoir_k = 32
    cfg.packing.scheduler_window = 8
    torch.manual_seed(0)
    eval_loader = GiftEvalEvalLoader.from_synthetic(cfg, n_items=cfg.eval.shard_windows)
    eval_log = []
    losses = run_training(cfg, steps=4, device="cpu",
                          eval_loader=eval_loader, eval_every=2, eval_log=eval_log)
    assert len(losses) == 4 and all(np.isfinite(losses))   # training path intact
    assert [s for s, _ in eval_log] == [2, 4]              # record-only eval fired
    assert all(np.isfinite(tl) for _, tl in eval_log)
