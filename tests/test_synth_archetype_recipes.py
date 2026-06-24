"""Tests for the archetype recipes + variety sampler (H1.1 'how to generate data')."""
import numpy as np

from tetris.data import synth_archetype_recipes as R


def test_every_recipe_generates_finite():
    for name, rec in R.RECIPES.items():
        C = len(rec["channels"])
        data = R.gen_from_recipe(np.random.default_rng(0), name, 3000, interval_min=10)
        assert data.shape == (C, 3000), name
        assert np.isfinite(data).all(), name


def test_recipe_period_sampling_decoupling():
    # the same recipe at a finer interval yields more samples-per-cycle (more, narrower
    # daily pulses) — period x sampling decoupling.
    hourly = R.gen_from_recipe(np.random.default_rng(0), "solar", 4000, interval_min=60)
    tenmin = R.gen_from_recipe(np.random.default_rng(0), "solar", 4000, interval_min=10)
    # count zero-crossings of the (mean-removed) series as a crude cycle proxy
    def cycles(x):
        z = x[0] - x[0].mean(); return int(np.sum((z[:-1] < 0) & (z[1:] >= 0)))
    assert cycles(tenmin) < cycles(hourly)               # 10-min: fewer, wider cycles in 4000 pts


def test_traffic_daily_recipes_seasonal_naive_learnable():
    # the daily-cycle traffic recipes (valley speed, broad-hump flow) must be learnable the
    # way the real data is: seasonal-naive clearly beats last-value (the daily profile
    # repeats). taxi_demand is a noise-dominated weak cycle (excluded — snaive only just
    # beats last there, by design).
    season = 24
    for name in ("traffic_speed", "traffic_flow"):
        x = R.gen_from_recipe(np.random.default_rng(0), name, 3000, interval_min=60)[0]
        ctx, y = x[:-season], x[-season:]
        scale = float(np.mean(np.abs(np.diff(ctx)))) + 1e-8
        last = np.mean(np.abs(ctx[-1] - y)) / scale
        snaive = np.mean(np.abs(ctx[-season:] - y)) / scale
        assert snaive < 0.8 * last, f"{name}: snaive {snaive:.2f} not < last {last:.2f}"


def test_m4_hourly_recipe_seasonal_naive_learnable():
    # m4_hourly is a clean daily cycle — seasonal-naive must clearly beat last-value,
    # the way it does on the real m4_hourly.
    season = 24
    x = R.gen_from_recipe(np.random.default_rng(0), "m4_hourly", 1000, interval_min=60)[0]
    ctx, y = x[:-season], x[-season:]
    scale = float(np.mean(np.abs(np.diff(ctx)))) + 1e-8
    last = np.mean(np.abs(ctx[-1] - y)) / scale
    snaive = np.mean(np.abs(ctx[-season:] - y)) / scale
    assert snaive < 0.8 * last


def test_solar_coarse_is_low_frequency_envelope_not_flat():
    # solar-D / solar-W are the coarse-resampled ANNUAL envelope — a low-frequency swing
    # (rendered as a partial-cycle sine), NOT a degenerate flat line and NOT high-frequency
    # noise. Guards the flat-panel regression: a long-correlation random drift could land flat
    # in a <1-cycle window, so the envelope is a sine that always swings.
    for name, iv in (("solar_daily", 1440), ("solar_weekly", 10080)):
        x = R.gen_from_recipe(np.random.default_rng(0), name, 6000, interval_min=iv)[0]
        tail = x[-300:]
        assert tail.std() > 0.1, f"{name}: degenerate near-flat output"
        # heavily smooth (kill the HF cloud noise) then count sign changes of the envelope —
        # a genuine low-frequency annual swing has only a handful over the window.
        z = tail - tail.mean()
        k = max(3, len(z) // 12)
        sm = np.convolve(z, np.ones(k) / k, mode="valid")
        crossings = int(np.sum((sm[:-1] < 0) & (sm[1:] >= 0)))
        assert 1 <= crossings <= 6, f"{name}: {crossings} envelope crossings — not low-freq"


def test_electricity_coarse_regime_blocks_have_quiet_and_active_stretches():
    # electricity-D / electricity-W are trapezoidal rise-stay-fall regime blocks over a
    # near-flat low baseline (business profile + regime suppression): there must be BOTH
    # elevated active stretches and quiet low stretches (not a uniform block train).
    for name, iv in (("electricity_daily", 1440), ("electricity_weekly", 10080)):
        x = R.gen_from_recipe(np.random.default_rng(3), name, 6000, interval_min=iv)[0]
        w = 40
        mu = np.array([x[i:i + w].mean() for i in range(0, len(x) - w, w)])
        # a clearly elevated stretch and a clearly low (quiet) stretch both occur
        assert mu.max() - mu.min() > 1.2, f"{name}: no active↔quiet regime contrast"


def test_gen_variety_finite_and_varied():
    fams = set()
    for i in range(30):
        data, meta = R.gen_variety(np.random.default_rng((1, i)), 3000)
        assert data.ndim == 2 and data.shape[1] == 3000
        assert np.isfinite(data).all()
        fams.add(meta["family"])
    assert len(fams) >= 3                                 # the sampler spans archetypes
