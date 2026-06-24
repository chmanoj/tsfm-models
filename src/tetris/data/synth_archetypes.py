"""Learnable-archetype synthetic generators (H1.1 rebuild) — data-driven.

Built from a *full-series, multi-timescale* characterization of the real GIFT-Eval
test data (not from a single stat). The finding: almost every **learnable** real
pattern is a **recurring profile** — a fixed within-period waveform that *repeats*
each period (which is exactly why seasonal-naive forecasts it), with day-to-day
amplitude variation, weekly modulation, and a small residual. The waveform shape is
the only thing that differs across datasets:

* zero-floored **pulse** — solar (night = flat zero, daytime bell);
* **business** profile — bizitobs (low night, daytime plateau, varying heights);
* load **double-hump** — electricity (morning + evening peaks, weekday/weekend);
* activity **hump + sparse spikes** — bitbrains.

Plus two non-daily learnable families: **trend/growth** (covid — linear/logistic) and a
multi-scale wrapper (a slow seasonal envelope modulating a daily profile — jena).

The core property we protect: the profile is *fixed and repeats*, so seasonal-naive
recovers it — i.e. the synth is **learnable** the way the real data is
([[learnable-structure-not-smooth-noise]]). Smooth noise is a *small residual*, never
the backbone. Everything is parametrized by **period-in-time × sampling interval**
([[synth-period-vs-sampling-frequency]]): a daily cycle is 144 samples at 10-min and 24
at hourly, so one generator produces solar/10T and solar/H.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import synthetic as S

# Recognized recurring-profile shapes.
PROFILE_KINDS = ("pulse", "business", "double_hump", "single_hump", "valley", "broad_hump")
GROWTH_KINDS = ("linear", "logistic", "exponential")


def _raised_bump(phase: np.ndarray, center: float, width: float) -> np.ndarray:
    """A Gaussian-ish bump on the *circular* phase axis (wraps at 1.0)."""
    d = np.abs((phase - center + 0.5) % 1.0 - 0.5)
    return np.exp(-0.5 * (d / max(1e-3, width)) ** 2)


def _flat_top(phase: np.ndarray, start: float, width: float, taper: float) -> np.ndarray:
    """A **trapezoid** (rise / stay / fall) on ``[start, start+width]``, zero elsewhere:
    a cosine ramp up over the first ``taper`` fraction of the window, a **flat plateau at
    1.0**, then a cosine ramp down over the last ``taper`` fraction. This is the
    rise-stay-fall shape real daily patterns have — not a sine bell."""
    local = (phase - start) / max(1e-6, width)
    inw = (local >= 0) & (local <= 1)
    lo = np.clip(local, 0.0, 1.0)
    t = max(1e-3, taper)
    rise = 0.5 * (1 - np.cos(np.pi * np.clip(lo / t, 0, 1)))
    fall = 0.5 * (1 - np.cos(np.pi * np.clip((1 - lo) / t, 0, 1)))
    return np.where(inw, np.minimum(rise, fall), 0.0)      # flat 1.0 between the tapers


def daily_profile(rng, spc: int, kind: str) -> np.ndarray:
    """One period's **fixed within-period waveform** (length ``spc`` = samples-per-cycle),
    standardized to unit scale with a non-negative floor where the archetype demands it.
    This is the shape that *repeats* — the learnable backbone. Profiles are **trapezoidal
    rise/stay/fall** (a flat plateau with defined rise/fall edges), not sine bells — real
    daily patterns ramp up, *hold*, then ramp down, rather than smoothly oscillating."""
    phase = np.linspace(0.0, 1.0, int(max(2, spc)), endpoint=False)
    if kind == "pulse":                                   # solar: zero night, daytime plateau
        # gradual sunrise/sunset tapers, flat midday plateau (cloud noise added later)
        prof = _flat_top(phase, rng.uniform(0.20, 0.30), rng.uniform(0.42, 0.56),
                         taper=rng.uniform(0.25, 0.40))
    elif kind == "business":                              # bizitobs: low night, day plateau
        # sharper on/off edges than solar, flat business-hours plateau, small floor
        prof = 0.05 + _flat_top(phase, rng.uniform(0.22, 0.32), rng.uniform(0.50, 0.66),
                                taper=rng.uniform(0.08, 0.18))
    elif kind == "double_hump":                           # electricity: morning + evening peaks
        a1, a2 = rng.uniform(0.5, 0.9), rng.uniform(0.8, 1.2)
        prof = (0.35 + a1 * _raised_bump(phase, rng.uniform(0.30, 0.40), 0.06)
                + a2 * _raised_bump(phase, rng.uniform(0.72, 0.82), 0.07))
    elif kind == "valley":                                # traffic SPEED: high plateau, rush dips
        # The inverse of double_hump — a high free-flow plateau (≈1) notched *down* by two
        # rush-hour congestion dips (LOOP_SEATTLE: AM + PM, asymmetric depth). Real daily
        # speed sits high almost all day and drops sharply twice; seasonal-naive recovers
        # the dip phases. Floored so it stays a high plateau, not a hump.
        # narrow + asymmetric dips so the high plateau dominates (real speed sits high
        # almost all day, then drops *sharply* — a small AM notch + a deep PM dip).
        d1, d2 = rng.uniform(0.15, 0.40), rng.uniform(0.60, 0.95)
        prof = (1.0 - d1 * _raised_bump(phase, rng.uniform(0.28, 0.36), 0.035)
                - d2 * _raised_bump(phase, rng.uniform(0.66, 0.80), 0.045))
        prof = np.clip(prof, 0.02, 1.0)
    elif kind == "broad_hump":                            # traffic flow: broad rounded day
        # a WIDE daytime elevation (~0.6 of the day) with gradual, rounded rise/fall (large
        # taper) and a short low night — M_DENSE. Unlike `business` (sharp square edges) the
        # big taper makes a soft trapezoid, not a square wave; unlike `single_hump` it is
        # wide (broad day, short night), matching the real day/night proportion.
        prof = 0.08 + _flat_top(phase, rng.uniform(0.12, 0.20), rng.uniform(0.58, 0.70),
                                taper=rng.uniform(0.34, 0.48))
    else:                                                 # single_hump: one daytime plateau
        prof = 0.12 + _flat_top(phase, rng.uniform(0.30, 0.45), rng.uniform(0.25, 0.45),
                                taper=rng.uniform(0.30, 0.5))
    return prof


def _weekly_factor(rng, n_days: int, *, weekend_low=(0.5, 0.85)) -> np.ndarray:
    """Per-day multiplier with a weekday/weekend contrast (the weekly modulation that
    makes weekly seasonal-naive meaningful). Weekday≈1, weekend lower."""
    wk = np.ones(7)
    wk[5:] = rng.uniform(*weekend_low, size=2)
    wk *= rng.uniform(0.9, 1.1, size=7)                   # mild day-of-week character
    return wk[np.arange(n_days) % 7]


def _regime_envelope(rng, n_days: int, switch_prob: float,
                     quiet=(0.08, 0.3)) -> np.ndarray:
    """A persistent **active↔quiet** regime multiplier over days: stays in a regime for a
    stretch, occasionally switching between full activity (1.0) and a quiet low-usage level
    — the busy-then-flat structure real load/usage series show (electricity meters going
    idle). Piecewise-constant (sharp transitions, as in the real data)."""
    env = np.ones(int(max(1, n_days))); state = 1.0
    for d in range(env.size):
        if rng.random() < switch_prob:
            state = float(rng.uniform(*quiet)) if state >= 0.99 else 1.0
        env[d] = state
    return env


def _persistent_amp(rng, n_days: int, jitter: float, persist: float) -> np.ndarray:
    """Per-day amplitude as an AR(1) around 1.0 — *autocorrelated* across days so the
    level persists (last-value forecasts well, as in real load data), while the daily
    profile still repeats (seasonal-naive forecasts well). iid jitter would break the
    persistence that real daily-cycle series have."""
    phi = float(np.clip(persist, 0.0, 0.98))
    innov = jitter * np.sqrt(max(1e-6, 1 - phi ** 2))
    a = np.empty(int(max(1, n_days))); a[0] = 1.0
    for d in range(1, a.size):
        a[d] = 1.0 + phi * (a[d - 1] - 1.0) + rng.normal(0, innov)
    return np.clip(a, 0.2, None)


def gen_recurring_profile(
    rng, n: int, spc: int, *, kind: Optional[str] = None, weekly: bool = True,
    amp_jitter: float = 0.18, amp_persist: float = 0.8, noise_amp: float = 0.04,
    hf_noise: float = 0.0, mult_noise: float = 0.0, level_frac: float = 0.0,
    regime_prob: float = 0.0, regime_quiet: Tuple[float, float] = (0.08, 0.3),
    trend: float = 0.0,
) -> Tuple[np.ndarray, int, str]:
    """A **recurring daily profile**: a fixed ``daily_profile`` repeated every ``spc``
    samples, scaled per day by an **autocorrelated** amplitude (``amp_persist`` →
    cross-day persistence + *different heights*, same shape), modulated weekly, blended
    with an optional **persistent slow level** (``level_frac`` — a smooth multi-day random
    walk that dominates a modest daily ripple, as in electricity load where last-value ≈
    seasonal-naive), plus a tiny smooth residual and an optional slow trend.

    Both baselines that work on the real data work here *by construction*: seasonal-naive
    (the profile repeats) **and** last-value (the level persists). Returns
    ``(series[n], season=spc, kind)``."""
    spc = int(max(2, spc))
    kind = kind or PROFILE_KINDS[int(rng.integers(len(PROFILE_KINDS)))]
    profile = daily_profile(rng, spc, kind)               # the fixed, repeating shape
    n_days = int(np.ceil(n / spc))
    day_amp = _persistent_amp(rng, n_days, amp_jitter, amp_persist)  # persistent heights
    if weekly:
        day_amp = day_amp * _weekly_factor(rng, n_days)
    if regime_prob > 0:                                  # active↔quiet usage regimes
        day_amp = day_amp * _regime_envelope(rng, n_days, regime_prob, quiet=regime_quiet)
    series = np.concatenate([day_amp[d] * profile for d in range(n_days)])[:n]
    if mult_noise > 0:
        # per-day intra-day texture (e.g. solar cloud cover): a sub-daily smooth factor
        # that varies day to day and occasionally dips to ~0, so the daytime "stay" phase
        # is noisy and sometimes falls to flat — not a clean bell. Multiplicative, so the
        # zero night stays zero. Clouds differ each day ⇒ not part of the repeating profile.
        factor = np.clip(1.0 + mult_noise * S._standardize(_smooth_resid(rng, n, corr=max(1.5, spc / 48))),
                         0.0, 1.4)
        series = series * factor
    lf = float(np.clip(level_frac, 0.0, 0.95))
    if lf > 0:                                            # persistent level + modest ripple
        # a long-correlation (multi-day), *non-integrated* level: it wanders over weeks
        # but is nearly flat over any single day, so last-value forecasts it well (as in
        # real electricity load, last-MASE 0.37). A random walk (cumsum) would drift too
        # much over a 24h horizon (~sqrt(H)) and break that persistence.
        level = S._standardize(_smooth_resid(rng, n, corr=max(2.0 * spc, 4.0)))
        series = (1 - lf) * S._standardize(series) + lf * level
    if trend:                                             # optional slow drift of the level
        series = series + trend * np.linspace(0, 1, n)
    series = S._standardize(series)
    # high-frequency (white) volatility — large hour-to-hour 1-step diffs on a persistent
    # level, as in real electricity load (it is what makes both last-value and
    # seasonal-naive read similar MASE there); separate from the smooth low-freq residual.
    out = series + hf_noise * rng.normal(0, 1, n) + noise_amp * S._standardize(_smooth_resid(rng, n))
    return out.astype(np.float64), spc, kind


def add_sparse_spikes(rng, x: np.ndarray, spc: int, *, rate_per_day=0.4,
                      amp=6.0) -> np.ndarray:
    """Add **recurring** sparse spikes (bitbrains): a few per several days, biased to the
    same daily phase so seasonal-naive still anticipates them (predictable, not random)."""
    n = x.size
    n_days = max(1, n // spc)
    phase0 = rng.uniform(0.3, 0.7)                         # consistent within-day phase
    sd = float(np.std(x)) + 1e-6
    for d in range(n_days):
        if rng.random() < rate_per_day:
            j = int((d + phase0) * spc + rng.integers(-spc // 20, spc // 20 + 1))
            if 0 <= j < n:
                x[j] += amp * sd * rng.uniform(0.8, 1.2)
    return x


def _smooth_resid(rng, n: int, corr: float = 4.0) -> np.ndarray:
    """Small smooth (low-frequency) residual — never the backbone, just texture."""
    if n < 2:
        return rng.normal(0, 1, n)
    # cap the half-width so the kernel never exceeds the series — otherwise np.convolve
    # mode="same" returns the (longer) kernel length, not n (a real bug at large corr).
    half = int(min(max(1, round(3 * corr)), max(1, (n - 1) // 2)))
    t = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (t / corr) ** 2); k /= k.sum()
    return np.convolve(rng.normal(0, 1, n), k, mode="same")


def gen_drift_seasonal(
    rng, n: int, spc: int, *, drift_corr_days: float = 8.0, weekly_amp: float = 0.0,
    daily_amp: float = 0.0, noise_amp: float = 0.12, hf_noise: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """The **multi-scale weather/drift** archetype (jena): a slow long-correlation drift
    backbone (the seasonal swing, which over a short forecast context reads as persistent
    drift — last-value/linear learnable) + an optional **weekly** recurring cycle
    (``weekly_amp`` — seasonal-naive-at-weekly learnable, as on jena humidity/pressure) +
    an optional small **daily** ripple + noise. Pure-drift channels set the cycle
    amplitudes to 0. Returns ``(series[n], weekly_season=7*spc)``.

    The backbone is *persistent* (smooth, long correlation), not a random walk that drifts
    away — so last-value forecasts the short horizon well, the way the real channels do."""
    spc = int(max(2, spc))
    drift = S._standardize(_smooth_resid(rng, n, corr=max(2.0, drift_corr_days * spc)))
    out = drift
    if weekly_amp > 0:                                    # fixed weekly profile, repeated
        wk_spc = 7 * spc
        wk_profile = S._standardize(_smooth_resid(rng, wk_spc, corr=max(2.0, wk_spc / 12)))
        reps = int(np.ceil(n / wk_spc))
        out = out + weekly_amp * np.tile(wk_profile, reps)[:n]
    if daily_amp > 0:                                     # daily oscillation UNDER the drift
        # a rounded daily cycle whose amplitude **varies day to day** (persistent AR(1)), so
        # it is a jittered modulation on top of the slow drift — not a clean constant-
        # amplitude sine (real ETT/weather daily cycles wax and wane across days).
        phase = 2 * np.pi * np.arange(n) / spc + rng.uniform(0, 2 * np.pi)
        n_days = int(np.ceil(n / spc))
        damp = _persistent_amp(rng, n_days, jitter=0.45, persist=0.7)
        out = out + daily_amp * np.repeat(damp, spc)[:n] * np.sin(phase)
    # smooth low-freq residual + optional WHITE high-frequency jitter (real ETT/weather
    # have genuine sample-to-sample noise; the smooth residual alone reads too clean).
    out = S._standardize(out) + noise_amp * S._standardize(_smooth_resid(rng, n, corr=2.0))
    if hf_noise > 0:
        out = out + hf_noise * rng.normal(0, 1, n)
    return out.astype(np.float64), 7 * spc


def gen_growth(rng, n: int, *, kind: Optional[str] = None, noise_amp: float = 0.04
               ) -> Tuple[np.ndarray, str]:
    """Clean **trend/growth** (covid): linear / logistic / exponential + small residual.
    Forecastable by a linear/last baseline — learnable, low-noise."""
    kind = kind or GROWTH_KINDS[int(rng.integers(len(GROWTH_KINDS)))]
    t = np.linspace(0, 1, n)
    if kind == "linear":
        g = t * rng.choice([-1.0, 1.0])
    elif kind == "logistic":
        # keep it **still rising at the horizon** (t0 late, gentle k) — real growth series
        # (covid) are mid-rise, so linear extrapolation beats last-value; a saturated
        # logistic would flatten and make last-value win, unlike the real data.
        t0 = rng.uniform(0.7, 1.0); k = rng.uniform(4, 8)
        g = 1.0 / (1.0 + np.exp(-k * (t - t0)))
    else:                                                 # exponential (accelerating)
        g = np.expm1(rng.uniform(1.5, 3.5) * t) / np.expm1(rng.uniform(1.5, 3.5))
    g = S._standardize(g)
    return (g + noise_amp * S._standardize(_smooth_resid(rng, n))).astype(np.float64), kind


def gen_channel(rng, n: int, spc: int, archetype: str, params: dict) -> np.ndarray:
    """Generate one channel from a named archetype (dispatch over the generators)."""
    p = dict(params or {})
    if archetype == "recurring":
        return gen_recurring_profile(rng, n, spc, **p)[0]
    if archetype == "drift_seasonal":
        return gen_drift_seasonal(rng, n, spc, **p)[0]
    if archetype == "growth":
        return gen_growth(rng, n, **p)[0]
    if archetype == "spikes":
        base = gen_drift_seasonal(rng, n, spc, weekly_amp=p.get("weekly_amp", 0.0),
                                  noise_amp=p.get("noise_amp", 0.05))[0]
        return add_sparse_spikes(rng, base, spc, rate_per_day=p.get("rate_per_day", 0.4),
                                 amp=p.get("amp", 6.0))
    raise ValueError(f"unknown archetype {archetype!r}")


def gen_multivariate(rng, n: int, spc: int, channel_specs, *, tie: float = 0.4
                     ) -> np.ndarray:
    """Compose a ``[C, n]`` multivariate series from a list of per-channel
    ``(archetype, params)`` specs, tied together by a **shared slow seasonal envelope**
    (``tie`` = how much of each channel co-moves with the common envelope — real
    multivariate weather/load channels share the season). ``tie=0`` ⇒ independent
    channels. The channel-archetype *assignment* is the only thing that differs between
    the targeted use (measured per real config) and the general use (sampled proportions);
    the composition is identical."""
    shared = S._standardize(_smooth_resid(rng, n, corr=max(2.0, 8.0 * spc)))  # common season
    tie = float(np.clip(tie, 0.0, 0.95))
    out = np.empty((len(channel_specs), n), dtype=np.float64)
    for c, (arch, params) in enumerate(channel_specs):
        x = S._standardize(gen_channel(rng, n, spc, arch, params))
        sign = 1.0 if rng.random() < 0.5 else -1.0        # channels co- or anti-move
        out[c] = S._standardize((1 - tie) * x + tie * sign * shared)
    return out


def samples_per_cycle(period_minutes: float, interval_minutes: float) -> int:
    """samples-per-cycle = period / sampling-interval (the period×frequency decoupling):
    a 1-day cycle (1440 min) is 144 samples at 10-min, 24 at hourly."""
    return int(max(2, round(period_minutes / max(1e-6, interval_minutes))))
