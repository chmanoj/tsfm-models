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


def test_gen_variety_finite_and_varied():
    fams = set()
    for i in range(30):
        data, meta = R.gen_variety(np.random.default_rng((1, i)), 3000)
        assert data.ndim == 2 and data.shape[1] == 3000
        assert np.isfinite(data).all()
        fams.add(meta["family"])
    assert len(fams) >= 3                                 # the sampler spans archetypes
