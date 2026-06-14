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
    for k, v in data.items():
        setattr(cfg.data, k, v)
    return cfg


def test_seasonal_naive_forecast_repeats_last_season():
    ctx = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # m=3 -> last season [3,4,5]
    fc = seasonal_naive_forecast(ctx, m=3, p=5)
    assert torch.equal(fc, torch.tensor([3.0, 4.0, 5.0, 3.0, 4.0]))


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
