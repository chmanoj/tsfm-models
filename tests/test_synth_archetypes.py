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


def test_profiles_are_trapezoids_not_bells():
    # rise/stay/fall: the on-region must have a genuine FLAT plateau (a run of samples at
    # the max), which a sine bell does not — only its single peak reaches the max.
    for kind in ("pulse", "business"):
        prof = A.daily_profile(np.random.default_rng(1), 288, kind)
        peak = prof.max()
        plateau_frac = float(np.mean(prof >= 0.98 * peak))
        assert plateau_frac > 0.10, f"{kind}: plateau {plateau_frac:.2f} too narrow (bell-like)"


def test_regime_shifts_produce_quiet_stretches():
    # active<->quiet regimes: a long quiet (low-amplitude) stretch must appear, and it is
    # a *persistent* stretch (not isolated low points), matching real busy/idle load.
    env = A._regime_envelope(np.random.default_rng(2), 400, switch_prob=0.04)
    assert env.min() < 0.4 and env.max() >= 0.99          # both regimes occur
    # the quiet regime persists: many consecutive low days, not single dips
    low = env < 0.5
    runs = np.diff(np.where(np.diff(np.r_[0, low.astype(int), 0]))[0])[::2]
    assert runs.size and runs.max() > 5


def test_growth_still_rising_linear_beats_last():
    # growth (covid) is still rising at the horizon ⇒ linear extrapolation beats a flat
    # last-value forecast (a saturated curve would wrongly favour last-value).
    for kind in A.GROWTH_KINDS:
        x, k = A.gen_growth(np.random.default_rng(0), 400, kind=kind)
        assert k == kind and np.isfinite(x).all()
        last, _, lin = _baselines(x, 12)
        assert lin < last


def test_drift_seasonal_weekly_vs_drift_character():
    # the weekly variant is forecastable by weekly seasonal-naive; the pure-drift variant
    # is not (last-value wins) — the two jena channel characters.
    def avg(weekly_amp, noise):
        ls, ss = [], []
        for k in range(4):
            x = A.gen_drift_seasonal(np.random.default_rng(k), 6000, 24,
                                     weekly_amp=weekly_amp, noise_amp=noise)[0]
            last, sn, _ = _baselines(x, 24 * 7)
            ls.append(last); ss.append(sn)
        return np.mean(ls), np.mean(ss)
    last_w, sn_w = avg(1.0, 0.05)
    assert sn_w < 0.7 * last_w                            # weekly cycle is learnable
    last_d, sn_d = avg(0.0, 0.1)
    assert last_d <= sn_d * 1.1                           # no weekly help; persistence wins


def test_multivariate_shape_and_shared_envelope():
    specs = [("drift_seasonal", dict(weekly_amp=0.7))] * 4 + [("spikes", dict())] * 2
    tied = A.gen_multivariate(np.random.default_rng(0), 4000, 24, specs, tie=0.6)
    indep = A.gen_multivariate(np.random.default_rng(0), 4000, 24, specs, tie=0.0)
    assert tied.shape == (6, 4000) and np.isfinite(tied).all()

    def offcorr(M):
        c = np.corrcoef(M); return float(np.nanmean(np.abs(c[np.triu_indices_from(c, 1)])))
    # the shared envelope makes tied channels co-move more than independent ones
    assert offcorr(tied) > offcorr(indep) + 0.1


def test_samples_per_cycle_decoupling():
    # a 1-day cycle is 144 samples at 10-min and 24 at hourly (period x sampling freq).
    assert A.samples_per_cycle(1440, 10) == 144
    assert A.samples_per_cycle(1440, 60) == 24
    assert A.samples_per_cycle(10080, 60) == 168    # weekly at hourly
