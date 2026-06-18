"""Tests for the learnable-archetype generators (H1.1 data-driven rebuild).

The defining property: the recurring-profile backbone is *learnable* — seasonal-naive
(repeat the last period) beats last-value, the way the real GIFT-Eval data is — and the
growth archetype is forecastable by a linear baseline. These guard against regressing
into smooth-random output that wins a smoothness stat but isn't forecastable
([[learnable-structure-not-smooth-noise]]).
"""
import numpy as np

from tetris.data import synth_archetypes as A


def _mase(pred, y, scale):
    return float(np.mean(np.abs(y - pred))) / scale


def _baselines(x, season):
    """(last-value MASE, seasonal-naive MASE, linear MASE) on a one-season tail horizon."""
    H = season
    ctx, y = x[:-H], x[-H:]
    scale = max(float(np.mean(np.abs(np.diff(ctx)))), 1e-6)
    last = _mase(np.full(H, ctx[-1]), y, scale)
    reps = int(np.ceil(H / season))
    snaive = _mase(np.tile(ctx[-season:], reps)[:H], y, scale)
    k = min(ctx.size, 3 * H); t = np.arange(k, dtype=float); tc = t - t.mean()
    slope = float(tc @ (ctx[-k:] - ctx[-k:].mean()) / (tc @ tc))
    lin = _mase(ctx[-1] + slope * np.arange(1, H + 1), y, scale)
    return last, snaive, lin


def test_recurring_profile_is_seasonal_naive_learnable():
    # for every profile shape, seasonal-naive must clearly beat last-value (the profile
    # repeats → it is learnable, by construction).
    for kind in A.PROFILE_KINDS:
        x, season, k = A.gen_recurring_profile(np.random.default_rng(0), 3000, 24,
                                               kind=kind, weekly=False)
        assert k == kind and season == 24 and np.isfinite(x).all()
        last, snaive, _ = _baselines(x, season)
        assert snaive < 0.8 * last, f"{kind}: snaive {snaive:.2f} not < last {last:.2f}"


def test_recurring_profile_persistence_and_determinism():
    # autocorrelated day amplitude ⇒ cross-day persistence (last-value is also decent);
    # and generation is deterministic for a fixed rng seed.
    a = A.gen_recurring_profile(np.random.default_rng(3), 2000, 24, kind="double_hump")[0]
    b = A.gen_recurring_profile(np.random.default_rng(3), 2000, 24, kind="double_hump")[0]
    assert np.array_equal(a, b)
    amp = A._persistent_amp(np.random.default_rng(1), 200, jitter=0.2, persist=0.85)
    assert np.corrcoef(amp[:-1], amp[1:])[0, 1] > 0.4   # consecutive days correlate


def test_pulse_profile_has_zero_floor():
    # the solar pulse must be genuinely zero outside the daytime window (flat night),
    # not a sine that dips negative.
    prof = A.daily_profile(np.random.default_rng(0), 144, "pulse")
    assert prof.min() >= -1e-9 and (prof <= 1e-6).mean() > 0.3   # a real off/night fraction


def test_growth_still_rising_linear_beats_last():
    # growth (covid) is still rising at the horizon ⇒ linear extrapolation beats a flat
    # last-value forecast (a saturated curve would wrongly favour last-value).
    for kind in A.GROWTH_KINDS:
        x, k = A.gen_growth(np.random.default_rng(0), 400, kind=kind)
        assert k == kind and np.isfinite(x).all()
        last, _, lin = _baselines(x, 12)
        assert lin < last


def test_samples_per_cycle_decoupling():
    # a 1-day cycle is 144 samples at 10-min and 24 at hourly (period x sampling freq).
    assert A.samples_per_cycle(1440, 10) == 144
    assert A.samples_per_cycle(1440, 60) == 24
    assert A.samples_per_cycle(10080, 60) == 168    # weekly at hourly
