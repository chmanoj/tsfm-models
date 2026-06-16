"""G5 — D13 phase-2 test-matched crop marginals (``auto_from_test_configs``).

CI-safe: the horizon set is supplied manually (the D13 manual override), so the
``sample_window`` horizon path, the progress-keyed reservoir crop schedule, and the
``_build_crop_schedule`` gating are all exercised without the real test table. The
live ``auto_from_test_configs`` derivation (lazy ``gift_eval`` + download) is covered
by the manual WSL run.
"""

from __future__ import annotations

import os
from math import ceil

import numpy as np
import pytest

from tetris.config import CurriculumCfg, SourceCfg, load_config
from tetris.packing.reservoir import StreamingReservoir, _build_crop_schedule
from tetris.tokenize.window_sampler import SamplerParams, sample_window

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
P_OUT = 16


def _params(**kw):
    base = dict(l_pack=4096, p_out=P_OUT, max_query_tokens=1 << 20)  # budget never binds
    base.update(kw)
    return SamplerParams(**base)


# --- sample_window horizon path ---------------------------------------------

def test_crop_horizons_pins_p_to_the_test_horizon():
    rng = np.random.default_rng(0)
    params = _params(crop_horizons=(48,))
    for _ in range(20):
        spec = sample_window(0, 1, t_raw=400, params=params, rng=rng)
        assert spec.p == 48 and ceil(spec.p / P_OUT) == spec.Q  # nt=1 -> Q=q_tok


def test_crop_horizon_clamped_to_series_and_budget():
    rng = np.random.default_rng(0)
    # budget cap: max_query_tokens=2 tokens -> 2*16 = 32 ceiling on p
    params = _params(crop_horizons=(96,), max_query_tokens=2)
    spec = sample_window(0, 1, t_raw=400, params=params, rng=rng)
    assert spec.p == 32                                    # clamped to the token budget
    # series cap: a short series bounds p to t_raw-1 (here below the token budget)
    short = _params(crop_horizons=(96,))
    spec2 = sample_window(0, 1, t_raw=10, params=short, rng=rng)
    assert spec2.p == 9                                    # t_raw-1, the binding cap


def test_no_crop_horizons_is_the_unchanged_broad_path():
    rng = np.random.default_rng(1)
    params = _params()                                     # crop_horizons None
    ps = {sample_window(0, 1, 400, params, rng).p for _ in range(50)}
    assert len(ps) > 1                                     # a distribution, not a single horizon


# --- _build_crop_schedule gating + switching --------------------------------

def _curric_cfg(**ckw):
    cfg = load_config(os.path.join(CONFIG_DIR, "streaming_synth.yaml"))
    cfg.data.loader = "curriculum"
    cfg.curriculum = CurriculumCfg(
        sources=[SourceCfg(name="s", loader="streaming", shard_root="x",
                           multiplier_phase1=1.0, multiplier_phase2=1.0)],
        total_items=1000, phase2_start=0.8, **ckw)
    return cfg


def test_schedule_off_for_non_curriculum_loader():
    cfg = load_config(os.path.join(CONFIG_DIR, "streaming_synth.yaml"))
    assert _build_crop_schedule(cfg, _params()) == (None, 0)


def test_schedule_off_when_distribution_none():
    cfg = _curric_cfg(phase2_crop_distribution="none")
    assert _build_crop_schedule(cfg, _params()) == (None, 0)


def test_schedule_switches_at_phase2_start():
    cfg = _curric_cfg(phase2_crop_distribution="auto_from_test_configs",
                      phase2_crop_horizons=[24, 48])
    sched, total = _build_crop_schedule(cfg, _params())
    assert total == 1000
    assert sched(0.0).crop_horizons is None                # broad in phase 1
    assert sched(0.79).crop_horizons is None
    assert sched(0.8).crop_horizons == (24, 48)            # test-matched from phase2_start
    assert sched(1.0).crop_horizons == (24, 48)


# --- reservoir honors the schedule (progress-keyed) -------------------------

class _LongSeries:
    """Infinite loader of identical long 1-channel series (so only the crop spec,
    not the data, drives p)."""

    def __iter__(self):
        import torch
        while True:
            yield (torch.zeros((1, 600), dtype=torch.float32), 0, 1)


def test_reservoir_applies_crop_schedule_by_progress():
    base = _params(l_pack=512, max_query_tokens=256)
    # p=48 -> q_tok=3, within the default q_tok_max=4 cap (4*16=64)
    test_matched = SamplerParams(l_pack=512, p_out=P_OUT, max_query_tokens=256,
                                 crop_horizons=(48,))
    # switch to test-matched once progress (items_pulled/crop_total) >= 0.5
    sched = lambda frac: test_matched if frac >= 0.5 else base
    res = StreamingReservoir(
        _LongSeries(), base, l_pack=512, p_out=P_OUT, buffers_per_step=2,
        reservoir_k=200, scheduler_window=8, crop_schedule=sched, crop_total=100)
    res._topup_reservoir()
    specs = [spec for (_it, spec) in res._reservoir]
    # first ~50 pulls (frac<0.5) are broad; pulls 50.. are pinned to p=48
    assert any(s.p != 48 for s in specs[:50])              # broad early
    assert all(s.p == 48 for s in specs[50:])              # test-matched late
