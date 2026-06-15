"""Experiment-tracker seam (G1).

Offline + dependency-free (no wandb, no network) so it runs in CI:
1. NullTracker is a total no-op; make_tracker degrades to it when disabled or when
   wandb can't be imported (the maintainer's online->offline->disabled chain);
2. mode resolution: WANDB_MODE wins, then login+reachability decide auto;
3. eval_scalars flattens an evaluate_mase dict / scalar and drops NaNs.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import types

from tetris.config import load_config
from tetris.train import tracking
from tetris.train.tracking import (
    NullTracker,
    WandbTracker,
    eval_scalars,
    make_tracker,
    _resolve_mode,
)


def _cfg(**tracking_kw):
    cfg = load_config("configs/sanity_sine.yaml")
    for k, v in tracking_kw.items():
        setattr(cfg.tracking, k, v)
    return cfg


def test_null_tracker_is_a_noop():
    t = NullTracker()
    t.log_config({"a": 1})
    t.log_scalars({"loss": 0.5}, step=3)
    t.finish()  # nothing raised, nothing returned


def test_make_tracker_disabled_returns_null():
    t = make_tracker(_cfg(backend="none"), run_name="r", run_dir="/tmp")
    assert isinstance(t, NullTracker)


def test_make_tracker_wandb_missing_degrades_to_null(monkeypatch, caplog):
    # Force `import wandb` to fail regardless of whether it's installed.
    monkeypatch.setitem(sys.modules, "wandb", None)
    # sanity_run.setup_logging may have set propagate=False on the tetris logger in
    # an earlier test; caplog hangs off the root, so force propagation to capture.
    monkeypatch.setattr(logging.getLogger("tetris"), "propagate", True)
    with caplog.at_level("WARNING", logger="tetris.tracking"):
        t = make_tracker(_cfg(backend="wandb"), run_name="r", run_dir="/tmp")
    assert isinstance(t, NullTracker)
    assert any("disabled" in m for m in caplog.messages)  # logged that it's off


def test_make_tracker_mode_disabled_returns_null(monkeypatch):
    # wandb importable but mode=disabled -> no run is created.
    monkeypatch.setitem(sys.modules, "wandb", types.ModuleType("wandb"))
    t = make_tracker(_cfg(backend="wandb", mode="disabled"), run_name="r", run_dir="/tmp")
    assert isinstance(t, NullTracker)


def test_make_tracker_wandb_builds_real_tracker(monkeypatch, tmp_path):
    # Fake wandb so we exercise the WandbTracker wrapper without a network/account.
    calls = {}

    class _Run:
        def __init__(self):
            self.config = types.SimpleNamespace(update=lambda c, **k: calls.setdefault("config", c))

        def log(self, scalars, step=None):
            calls.setdefault("logs", []).append((step, scalars))

        def finish(self):
            calls["finished"] = True

    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: (calls.setdefault("init", kw), _Run())[1]
    monkeypatch.setitem(sys.modules, "wandb", fake)

    t = make_tracker(_cfg(backend="wandb", mode="offline"),
                     run_name="run-1", run_dir=tmp_path, config={"x": 1})
    assert isinstance(t, WandbTracker)
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["name"] == "run-1"
    assert calls["init"]["config"] == {"x": 1}
    t.log_scalars({"loss": 0.25}, step=7)
    t.finish()
    assert calls["logs"] == [(7, {"loss": 0.25})]
    assert calls["finished"] is True


def test_resolve_mode_env_overrides(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert _resolve_mode("auto") == "offline"
    assert _resolve_mode("online") == "offline"  # env wins over explicit too


def test_resolve_mode_auto_offline_when_not_logged_in(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(tracking, "_logged_in", lambda: False)
    monkeypatch.setattr(tracking, "_reachable", lambda *a, **k: True)
    assert _resolve_mode("auto") == "offline"


def test_resolve_mode_auto_online_when_logged_in_and_reachable(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(tracking, "_logged_in", lambda: True)
    monkeypatch.setattr(tracking, "_reachable", lambda *a, **k: True)
    assert _resolve_mode("auto") == "online"


def test_resolve_mode_explicit_passthrough(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert _resolve_mode("online") == "online"
    assert _resolve_mode("offline") == "offline"


def test_eval_scalars_flattens_dict_and_drops_nan():
    out = eval_scalars({"model_mase": 0.8, "snaive_mase": 1.0, "skill": float("nan"),
                        "n": 12, "skipped": 0})
    assert out == {"eval/model_mase": 0.8, "eval/snaive_mase": 1.0,
                   "eval/n": 12.0, "eval/skipped": 0.0}
    assert "eval/skill" not in out  # NaN dropped


def test_eval_scalars_scalar_loss():
    out = eval_scalars(0.5)
    assert out == {"eval/test_loss": 0.5}
    assert math.isclose(out["eval/test_loss"], 0.5)


def test_load_dotenv_sets_missing_without_override(monkeypatch, tmp_path):
    from tetris.train.tracking import _load_dotenv

    env = tmp_path / ".env"
    env.write_text('# comment\nWANDB_API_KEY="from-dotenv"\nALREADY_SET=new\nBLANK\n')
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("ALREADY_SET", "old")
    _load_dotenv(str(env))
    assert os.environ["WANDB_API_KEY"] == "from-dotenv"  # quotes stripped, set
    assert os.environ["ALREADY_SET"] == "old"            # pre-set var not overridden


def test_load_dotenv_missing_file_is_noop():
    from tetris.train.tracking import _load_dotenv

    _load_dotenv("/no/such/.env")  # must not raise
