"""G3.1 — max-query-token cap + iterative rollout.

Reproduces (and pins the fix for) the eval query-token overflow that blocked G3:
medium/long GIFT-Eval terms (wide multivariate × long horizon) sized the per-segment
query budget ``Q_total = n_horizon_channels·ceil(p/p_out)`` above ``L_pack``, so the
context budget clamped to 1 and the segment overflowed the buffer
(``ValueError: buffer 0 overflow ... 966 > 512``).

The fix: a first-class ``packing.max_query_tokens`` budget. A single forward pass
predicts only ``p_pred`` steps (what fits the budget); the **full** benchmark horizon
is covered by an autoregressive :func:`rollout_forecast`. Cap-off behaviour does not
exist — the budget is always real, clamped to ``L_pack − C`` so a context token fits.

Everything here is pure tokenizer/packing math + a tiny random-init model, so it runs
offline on Mac (no GIFT-Eval data, no ``gift_eval``, no network) — the recipe from
``prompts/gifteval_G3.1.md``.
"""

import math
import os

import torch

from tetris.config import load_config
from tetris.constants import Role
from tetris.data.contract import EvalItem
from tetris.data.eval_loader import (
    _capped_p,
    eval_batch,
    rollout_forecast,
)
from tetris.model.tetris import Tetris
from tetris.tokenize.window_sampler import SamplerParams, sample_window

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

# jena_weather/long shape — the worst real overflow (C=21, p=720, q_tok=45 uncapped
# -> Q_total=945 >> L_pack=512). See the G3 reconciliation block.
C, T_CTX, P, P_OUT, L = 21, 256, 720, 16, 512


def _wide_long_item():
    return EvalItem(
        data_tensor=torch.randn(C, T_CTX),
        num_features=0,
        num_targets=C,
        y_true=torch.randn(P, C),
        naive_denom=None,
        config_id="repro/long",
        season_length=1,
    )


def test_capped_p_keeps_segment_within_lpack():
    """``_capped_p`` bounds the per-pass query budget so ``Q_total ≤ L_pack − C``."""
    params = SamplerParams(l_pack=L, p_out=P_OUT, max_query_tokens=128)
    p_pred = _capped_p(P, C, C, params, L)        # n_horizon == C (all targets)
    q_tok = math.ceil(p_pred / P_OUT)
    assert p_pred < P                              # genuinely truncated
    assert C * q_tok <= 128                        # within the configured budget
    assert C * q_tok <= L - C                      # ... and leaves a context token


def test_eval_batch_no_overflow_with_default_params():
    """The prompt's exact repro recipe: with default params (no explicit cap) the
    runtime clamp to ``L_pack − C`` alone already prevents the old overflow."""
    item = _wide_long_item()
    params = SamplerParams(l_pack=L, p_out=P_OUT, tier_prior=(16, 16, 16, 16, 16, 8))
    batch = eval_batch(item, params, p_out=P_OUT, l_pack=L)   # used to raise ValueError(966 > 512)
    # one segment, packed into one buffer that fits L_pack
    used = int((batch.sample_id[0] >= 0).sum())
    assert used <= L


def test_eval_batch_truncates_to_budget():
    """With the gifteval 128-token budget the single pass predicts only p_pred steps."""
    item = _wide_long_item()
    params = SamplerParams(l_pack=L, p_out=P_OUT, max_query_tokens=128)
    batch = eval_batch(item, params, p_out=P_OUT, l_pack=L)
    n_qry = int((batch.role[0] == int(Role.QRY)).sum())
    p_pred = _capped_p(P, C, C, params, L)
    assert n_qry == C * math.ceil(p_pred / P_OUT)            # query slots == capped budget
    assert int((batch.sample_id[0] >= 0).sum()) <= L


def _tiny_model_cfg():
    cfg = load_config(os.path.join(CONFIG_DIR, "gifteval_test_overfit.yaml"))
    cfg.model.d_model = 16
    cfg.model.n_layers = 1
    cfg.model.n_heads = 2
    cfg.model.out_patch = P_OUT
    cfg.packing.L_pack = L
    cfg.packing.max_query_tokens = 128
    return cfg


def test_rollout_covers_full_horizon(monkeypatch):
    """``rollout_forecast`` returns a forecast for the **whole** horizon (all finite)
    by iterating ceil(p/p_pred) capped passes — a single pass would leave NaNs."""
    import tetris.data.eval_loader as EL

    cfg = _tiny_model_cfg()
    torch.manual_seed(0)
    model = Tetris(cfg)
    item = _wide_long_item()
    params = SamplerParams(l_pack=L, p_out=P_OUT, max_query_tokens=128)

    passes = {"n": 0}
    real = EL._forward_forecast

    def _counting(*a, **k):
        passes["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(EL, "_forward_forecast", _counting)

    fc = rollout_forecast(model, item, params, d_model=cfg.model.d_model, p_out=P_OUT,
                          l_pack=L, device="cpu", basis_seed=0)
    p_pred = _capped_p(P, C, C, params, L)
    assert fc.shape == (P, C)
    assert torch.isfinite(fc).all()                          # every step covered
    assert passes["n"] == math.ceil(P / p_pred) > 1          # actually iterated


def test_rollout_single_pass_for_small_horizon(monkeypatch):
    """A horizon already within budget rolls out in exactly one pass (the cap-off
    path is byte-for-byte the old single-shot eval)."""
    import tetris.data.eval_loader as EL

    cfg = _tiny_model_cfg()
    torch.manual_seed(0)
    model = Tetris(cfg)
    short = EvalItem(torch.randn(3, 128), 0, 3, torch.randn(P_OUT, 3), None,
                     "repro/short", season_length=1)
    params = SamplerParams(l_pack=L, p_out=P_OUT, max_query_tokens=128)

    passes = {"n": 0}
    real = EL._forward_forecast
    monkeypatch.setattr(EL, "_forward_forecast",
                        lambda *a, **k: (passes.__setitem__("n", passes["n"] + 1), real(*a, **k))[1])

    fc = rollout_forecast(model, short, params, d_model=cfg.model.d_model, p_out=P_OUT,
                          l_pack=L, device="cpu", basis_seed=0)
    assert fc.shape == (P_OUT, 3)
    assert passes["n"] == 1


def test_inversion_clamp_keeps_raw_finite():
    """A pathological arcsinh-space prediction (1000) must NOT overflow to inf:
    the ±ARCSINH_INV_CLAMP guard caps sinh so raw space stays finite (G3.1)."""
    from types import SimpleNamespace
    import tetris.normalize as Nrm
    from tetris.data.eval_loader import horizon_forecast_raw

    p_out, p, nt = 2, 2, 1
    out = SimpleNamespace(horizon=torch.full((1, 1, p_out), 1000.0))   # [B,L,P_out], huge
    batch = SimpleNamespace(
        role=torch.tensor([[int(Role.QRY)]]),
        channel_idx=torch.tensor([[0]]),
        t_center=torch.tensor([[p_out * 0.5]]),
        stats_a=torch.tensor([[0.0]]),
        stats_sigma=torch.tensor([[100.0]]),
    )
    item = EvalItem(torch.zeros(1, 4), 0, nt, torch.zeros(p, nt), None, "x", season_length=1)
    fc = horizon_forecast_raw(out, batch, item, p_out=p_out)
    assert torch.isfinite(fc).all()                           # sinh(1000) would be inf
    # clamped value is sinh(10)*100 ~ 1.1e6 (not inf)
    assert fc.abs().max() < 2.0e6 and fc.abs().max() > 1.0e5
    assert Nrm.ARCSINH_INV_CLAMP == 10.0


def test_rollout_sanitizes_nonfinite_feedback(monkeypatch):
    """If a rollout pass emits non-finite predictions, they are NOT fed back into the
    next pass's context (sanitized) — so one bad pass can't cascade NaN (G3.1)."""
    import tetris.data.eval_loader as EL

    cfg = _tiny_model_cfg()
    torch.manual_seed(0)
    model = Tetris(cfg)
    item = _wide_long_item()                                  # long -> several passes
    params = SamplerParams(l_pack=L, p_out=P_OUT, max_query_tokens=128)

    calls = {"n": 0, "ctx_finite": []}

    def fake_fwd(forward, sub, params_, **k):
        calls["ctx_finite"].append(bool(torch.isfinite(sub.data_tensor).all()))
        calls["n"] += 1
        chunk = int(sub.y_true.shape[0])
        out = torch.zeros(chunk, sub.num_targets)
        if calls["n"] == 1:
            out[:] = float("inf")                             # pathological first pass
        return out

    monkeypatch.setattr(EL, "_forward_forecast", fake_fwd)
    EL.rollout_forecast(model, item, params, d_model=cfg.model.d_model, p_out=P_OUT,
                        l_pack=L, device="cpu", basis_seed=0)
    assert calls["n"] >= 2
    assert all(calls["ctx_finite"][1:])   # pass 2+ context finite despite pass-1 inf


def test_training_sampler_respects_cap():
    """``sample_window`` bounds the sampled horizon so a training segment never
    one-shots a huge horizon: ``Q_total ≤ max_query_tokens``."""
    params = SamplerParams(l_pack=L, p_out=P_OUT, q_tok_max=64, max_query_tokens=128)
    rng = __import__("numpy").random.default_rng(0)
    for _ in range(50):
        spec = sample_window(0, C, T_CTX + P, params, rng)   # wide, long-enough series
        assert spec.Q_total <= 128
        assert spec.S <= L
