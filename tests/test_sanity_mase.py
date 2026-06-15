"""Sanity stage (simple-synthetic train->test, scored vs seasonal naive).

Offline + deterministic (no network / no optional deps), so it runs in CI:
1. the MASE helpers (gluonts/GIFT-Eval formula) on hand-checkable inputs;
2. the matched sanity train/eval loaders honor the frozen Item/EvalItem contract;
3. the full pipeline runs and the model *learns* a clean sine — final MASE beats
   its random-init MASE (capacity smoke, a few steps, tiny model).
"""

from __future__ import annotations

import math

import torch

from tetris.config import load_config
from tetris.data.contract import build_eval_loader, build_loader, validate_item
from tetris.data.eval_loader import evaluate_mase
from tetris.metrics import mase, seasonal_naive_denom, seasonal_naive_forecast
from tetris.train.sanity_run import run_sanity


def _tiny_cfg(tmp_path, **data):
    cfg = load_config("configs/sanity_sine.yaml")
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.packing.L_pack = 128
    cfg.packing.buffers_per_step = 2
    cfg.packing.reservoir_k = 16
    cfg.packing.scheduler_window = 16
    cfg.data.n_series = 8
    cfg.eval.shard_windows = 8
    cfg.tracking.backend = "none"   # offline/CI: never touch wandb in tests
    for k, v in data.items():
        setattr(cfg.data, k, v)
    return cfg


def test_seasonal_naive_forecast_repeats_last_season():
    ctx = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # m=3 -> last season [3,4,5]
    fc = seasonal_naive_forecast(ctx, m=3, p=5)
    assert torch.equal(fc, torch.tensor([3.0, 4.0, 5.0, 3.0, 4.0]))


def test_seasonal_naive_forecast_imputes_missing_like_gluonts():
    """Missing values in the seasonal lag are last-value-imputed (gluonts
    SeasonalNaivePredictor) so the baseline never emits NaN (G3.1)."""
    nan = float("nan")
    # last season [3, NaN, 5] -> forward-fill -> [3, 3, 5]; tiled over p=5
    ctx = torch.tensor([0.0, 1.0, 2.0, 3.0, nan, 5.0])
    fc = seasonal_naive_forecast(ctx, m=3, p=5)
    assert torch.isfinite(fc).all()
    assert torch.equal(fc, torch.tensor([3.0, 3.0, 5.0, 3.0, 3.0]))
    # leading NaN -> first finite value (back-fill)
    lead = seasonal_naive_forecast(torch.tensor([nan, nan, 4.0]), m=3, p=3)
    assert torch.equal(lead, torch.tensor([4.0, 4.0, 4.0]))
    # all-NaN context -> 0 (DummyValueImputation default), still finite
    allnan = seasonal_naive_forecast(torch.tensor([nan, nan, nan, nan]), m=2, p=4)
    assert torch.isfinite(allnan).all() and torch.equal(allnan, torch.zeros(4))


def test_seasonal_naive_denom_mean_abs_seasonal_diff():
    ctx = torch.tensor([1.0, 2.0, 4.0, 7.0])  # m=1 diffs |1|,|2|,|3| -> mean 2
    assert math.isclose(float(seasonal_naive_denom(ctx, m=1)), 2.0, rel_tol=1e-6)


def test_mase_is_mae_over_denom():
    y = torch.tensor([1.0, 2.0, 3.0])
    yhat = torch.tensor([1.5, 2.5, 3.5])  # MAE 0.5
    denom = torch.tensor(0.25)
    assert math.isclose(mase(y, yhat, denom), 2.0, rel_tol=1e-6)


def test_sanity_loaders_honor_contract(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    train = build_loader(cfg)
    it = iter(train)
    for _ in range(4):
        item = next(it)
        validate_item(item)
        assert item[0].shape[1] == cfg.data.series_len - cfg.data.horizon
    eval_loader = build_eval_loader(cfg)
    assert len(eval_loader) == cfg.data.n_series
    e0 = eval_loader[0]
    assert e0.season_length == cfg.data.season_lengths[0]
    assert e0.y_true.shape[0] == cfg.data.horizon


def test_sanity_pipeline_learns_sine(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    model = None
    from tetris.model.tetris import Tetris

    torch.manual_seed(0)
    model = Tetris(cfg).to("cpu")
    eval_loader = build_eval_loader(cfg)
    base = evaluate_mase(model, eval_loader, cfg, device="cpu")
    assert math.isfinite(base["model_mase"]) and math.isfinite(base["snaive_mase"])
    assert base["n"] == cfg.data.n_series

    from tetris.train.loop import run_training

    run_training(cfg, steps=120, lr=1e-3, device="cpu", model=model)
    after = evaluate_mase(model, eval_loader, cfg, device="cpu")
    # The model should learn *something* — horizon MASE drops well below random init.
    assert after["model_mase"] < base["model_mase"]


def test_kff_eval_reveals_feature_future_without_leaking_targets(tmp_path):
    cfg = _tiny_cfg(tmp_path, case="features_target", n_channels=2,
                    known_future_features=True)
    eval_loader = build_eval_loader(cfg)
    e0 = eval_loader[0]
    assert e0.num_features == 1 and e0.num_targets == 1
    # KFF: feature future is revealed, shaped [p, num_features]
    assert e0.feature_future is not None
    assert tuple(e0.feature_future.shape) == (cfg.data.horizon, e0.num_features)
    # the context itself never contains the held-out horizon (no target leak)
    assert e0.data_tensor.shape[1] == cfg.data.series_len - cfg.data.horizon

    # non-KFF default keeps the feature future hidden
    cfg2 = _tiny_cfg(tmp_path, case="features_target", n_channels=2)
    assert build_eval_loader(cfg2)[0].feature_future is None


def test_kff_tokens_are_built_for_known_future_features(tmp_path):
    # eval_spec with kff_features marks the feature channels KFF and allocates
    # K = n_features * q_tok future tokens (the bug fix: features were past-only).
    from tetris.data.eval_loader import eval_batch, eval_spec
    from tetris.tokenize.window_sampler import SamplerParams
    from tetris.constants import ContentState, Role

    cfg = _tiny_cfg(tmp_path, case="kff_driver", n_channels=2,
                    known_future_features=True)
    params = SamplerParams(l_pack=cfg.packing.L_pack, p_out=cfg.model.out_patch,
                           tier_prior=tuple(cfg.model.tier_alloc_per_channel))
    spec = eval_spec(1, 1, 100, cfg.data.horizon, params, kff_features=True)
    assert spec.K == 1 * spec.q_tok and spec.Q_total == spec.Q + spec.K

    item = build_eval_loader(cfg)[0]
    b = eval_batch(item, params, p_out=cfg.model.out_patch, l_pack=cfg.packing.L_pack)
    role, chan, tc, cs = b.role[0], b.channel_idx[0], b.t_center[0], b.content_state[0]
    # feature channel 0 has OBSERVED CTX tokens at t_center>0 == the KFF future tokens
    kff = (chan == 0) & (role == int(Role.CTX)) & (tc > 0) & (cs == int(ContentState.OBSERVED))
    assert int(kff.sum()) == spec.q_tok


def test_run_sanity_writes_artifacts(tmp_path, monkeypatch):
    import tetris.train.sanity_run as sr

    monkeypatch.setattr(sr, "OUTPUTS_ROOT", tmp_path)
    # keep it fast: patch the config loader to the tiny config
    real_load = sr.load_config

    def _tiny(path):
        return _tiny_cfg(tmp_path)

    monkeypatch.setattr(sr, "load_config", _tiny)
    sr.run_sanity("configs/sanity_sine.yaml", steps=20, eval_every=10, n_plot=2)
    runs = list(tmp_path.glob("sanity_sine_*"))
    assert runs, "run dir not created"
    rd = runs[0]
    for name in ("command.txt", "config.yaml", "train_log.txt", "samples.png"):
        assert (rd / name).exists(), f"missing {name}"
    assert real_load is not None  # silence lint on the captured ref
