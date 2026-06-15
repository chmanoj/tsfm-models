"""G2 — real GIFT-Eval data access + leaderboard MASE.

Two postures, mirroring ``test_eval_loss``:
1. The **leaderboard aggregator** (geo-mean MASE across configs + per-config
   breakdown + per-config item cap) is exercised entirely **offline** on a
   hand-built multi-config ``EvalItem`` shard with a tiny real model — CI-safe.
2. The **real download/iterators** (``download_gifteval`` / ``iter_eval_items`` /
   ``iter_train_items``) touch the network + optional ``gift_eval``/``gluonts``
   deps, so those tests **skip when the deps/data are absent** and never make CI
   depend on the network. The storage-root guard (no env, no arg) is dep-free and
   always checked.
"""

from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest
import torch

from tetris.config import load_config
from tetris.data.contract import EvalItem
from tetris.data.eval_loader import GiftEvalEvalLoader, evaluate_leaderboard, _gmean
from tetris.data import gifteval_download as gd
from tetris.metrics import mase, seasonal_naive_denom

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
P_OUT = 8

_HAS_GLUONTS = importlib.util.find_spec("gluonts") is not None
_HAS_GIFT_EVAL = importlib.util.find_spec("gift_eval") is not None
_HAS_HF = importlib.util.find_spec("huggingface_hub") is not None


def _cfg():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.packing.L_pack = 256
    cfg.packing.buffers_per_step = 2
    cfg.model.out_patch = P_OUT
    cfg.model.d_model = 16
    cfg.model.n_layers = 1
    cfg.model.n_heads = 2
    cfg.tracking.backend = "none"
    return cfg


def _items(n_configs=3, per_config=4, season=8):
    """A multi-config EvalItem shard: deterministic sine windows, grouped by
    ``config_id`` with a dataset season — the leaderboard's input contract."""
    items = []
    for c in range(n_configs):
        for w in range(per_config):
            t_ctx = 64
            x = np.sin((np.arange(t_ctx + P_OUT) + 3 * c + w) * 0.3).astype(np.float32)[None, :]
            context = torch.from_numpy(x[:, :t_ctx].copy())
            y_true = torch.from_numpy(x[:, t_ctx:].T.copy())  # [p, 1]
            items.append(EvalItem(
                data_tensor=context, num_features=0, num_targets=1,
                y_true=y_true, naive_denom=None, config_id=f"cfg/{c}",
                season_length=season,
            ))
    return items


def _model(cfg):
    from tetris.model.tetris import Tetris

    torch.manual_seed(0)
    return Tetris(cfg).to("cpu")


# --- 1. offline leaderboard aggregator -----------------------------------------

def test_leaderboard_geomean_across_configs():
    cfg = _cfg()
    cfg.eval.items_per_config = -1
    loader = GiftEvalEvalLoader(_items(n_configs=3, per_config=4))
    model = _model(cfg)
    res = evaluate_leaderboard(model, loader, cfg, device="cpu")

    assert res["n_configs"] == 3
    assert set(res["per_config"]) == {"cfg/0", "cfg/1", "cfg/2"}
    assert res["skipped"] == 0
    # The leaderboard score is the geometric mean of the per-config MASEs.
    per = [res["per_config"][c]["model_mase"] for c in sorted(res["per_config"])]
    assert all(math.isfinite(v) for v in per)
    expected = float(np.exp(np.mean(np.log(np.clip(per, 1e-12, None)))))
    assert math.isclose(res["leaderboard_mase"], expected, rel_tol=1e-6)
    assert math.isfinite(res["snaive_mase"]) and math.isfinite(res["skill"])


def test_leaderboard_caps_items_per_config():
    cfg = _cfg()
    loader = GiftEvalEvalLoader(_items(n_configs=2, per_config=5))
    model = _model(cfg)

    cfg.eval.items_per_config = 2
    capped = evaluate_leaderboard(model, loader, cfg, device="cpu")
    assert all(v["n_items"] == 2 for v in capped["per_config"].values())

    cfg.eval.items_per_config = -1
    full = evaluate_leaderboard(model, loader, cfg, device="cpu")
    assert all(v["n_items"] == 5 for v in full["per_config"].values())

    # explicit override beats the config value
    over = evaluate_leaderboard(model, loader, cfg, device="cpu", items_per_config=1)
    assert all(v["n_items"] == 1 for v in over["per_config"].values())


def test_leaderboard_skips_items_without_season():
    cfg = _cfg()
    cfg.eval.items_per_config = -1
    items = _items(n_configs=2, per_config=2)
    # blank out the season on one config -> those items are skipped, not scored
    items = [e._replace(season_length=None) if e.config_id == "cfg/0" else e for e in items]
    res = evaluate_leaderboard(_model(cfg), GiftEvalEvalLoader(items), cfg, device="cpu")
    assert res["n_configs"] == 1 and "cfg/1" in res["per_config"]
    assert res["skipped"] == 2


# --- model-NaN poisons (guard) vs data-NaN masked (gluonts posture) -------------

def test_gmean_poisons_on_model_nonfinite_not_on_empty():
    # finite geo-mean
    assert math.isclose(_gmean([1.0, 4.0]), 2.0, rel_tol=1e-9)
    # a model NaN/inf must NOT be hidden — it poisons the aggregate (the guard)
    assert math.isinf(_gmean([2.0, float("inf")]))
    assert math.isnan(_gmean([1.0, float("nan")]))
    # empty input (a pure data reason: nothing scorable) -> NaN, not poison
    assert math.isnan(_gmean([]))


def test_model_forecast_is_not_masked_so_nonfinite_poisons():
    # The forecast is passed to mase() unmasked: a model inf at a data-valid step
    # propagates to MASE (we never silently drop it).
    yt = torch.tensor([1.0, 2.0, 3.0])
    fc = torch.tensor([1.0, float("inf"), 3.0])
    denom = torch.tensor(1.0)
    assert math.isinf(mase(yt, fc, denom))


def test_seasonal_naive_denom_masks_data_nans():
    # Data NaNs in the context are masked out of the denom (gluonts masked_invalid),
    # never producing a NaN denom — this is *data*, not model, handling.
    ctx = torch.tensor([1.0, 2.0, float("nan"), 4.0, 7.0])  # m=1 finite diffs |1|,|3|
    assert math.isclose(float(seasonal_naive_denom(ctx, m=1)), 2.0, rel_tol=1e-6)
    allnan = torch.tensor([float("nan"), float("nan")])
    assert math.isfinite(float(seasonal_naive_denom(allnan, m=1)))  # floored, not NaN


def test_leaderboard_data_nan_horizon_is_masked_not_skipped():
    # An item whose held-out target has *some* NaN steps is still scored on the
    # finite steps (masked), as long as the model output is finite.
    cfg = _cfg()
    cfg.eval.items_per_config = -1
    items = _items(n_configs=1, per_config=2)
    yt = items[0].y_true.clone()
    yt[0, 0] = float("nan")  # one missing horizon observation (data)
    items[0] = items[0]._replace(y_true=yt)
    res = evaluate_leaderboard(_model(cfg), GiftEvalEvalLoader(items), cfg, device="cpu")
    # config still scored (data NaN masked, not treated as model failure or skip)
    assert res["n_configs"] == 1 and res["skipped"] == 0
    assert res["per_config"]["cfg/0"]["n_channel_items"] == 2


# --- 2. real iterators: dep-free guard + skipped-when-absent --------------------

def test_real_iterators_need_storage_root(monkeypatch, tmp_path):
    """Dep-free: no local_dir + no $GIFT_EVAL (+ no .env) -> ValueError before any
    lazy import. chdir to a tmp dir AND neutralize python-dotenv so an ambient
    repo-root .env (e.g. the WSL host's, which defines GIFT_EVAL) can't supply the
    path — python-dotenv discovers .env from the source-file location, not cwd, so
    chdir alone isn't enough on every host."""
    monkeypatch.delenv(gd.TEST_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    except ImportError:  # dotenv absent -> nothing to neutralize
        pass
    with pytest.raises(ValueError, match="storage root"):
        next(gd.iter_eval_items(""))
    with pytest.raises(ValueError, match="storage root"):
        next(gd.iter_train_items(""))


@pytest.mark.skipif(_HAS_HF, reason="huggingface_hub installed; ImportError path not taken")
def test_download_requires_hf_when_absent():
    with pytest.raises(ImportError, match="huggingface_hub"):
        gd.download_gifteval("/tmp/nope")


@pytest.mark.skipif(not _HAS_GLUONTS, reason="needs gluonts for get_seasonality")
def test_season_length_is_gifteval_source():
    # GIFT-Eval derives MASE seasonality from the frequency via gluonts (non-silent).
    assert gd._season_length("H") == 24
    assert gd._season_length("M") == 12
    assert gd._season_length("D") == 1
    assert gd._season_length("T") == 1440


@pytest.mark.skipif(
    not (_HAS_GIFT_EVAL and _HAS_GLUONTS and os.getenv(gd.TEST_ENV_VAR)),
    reason="needs gift_eval + gluonts + a populated $GIFT_EVAL tree (network/data)",
)
def test_iter_eval_items_smoke_on_real_tree():  # pragma: no cover - needs real data
    items = list(gd.iter_eval_items(items_per_config=2))
    assert items, "no GIFT-Eval items found under $GIFT_EVAL"
    e = items[0]
    assert e.num_targets >= 1 and e.season_length is not None
    assert e.data_tensor.dim() == 2 and e.y_true.dim() == 2
