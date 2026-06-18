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
PROFILE_KINDS = ("pulse", "business", "double_hump", "single_hump")
GROWTH_KINDS = ("linear", "logistic", "exponential")


def _raised_bump(phase: np.ndarray, center: float, width: float) -> np.ndarray:
    """A Gaussian-ish bump on the *circular* phase axis (wraps at 1.0)."""
    d = np.abs((phase - center + 0.5) % 1.0 - 0.5)
    return np.exp(-0.5 * (d / max(1e-3, width)) ** 2)


def daily_profile(rng, spc: int, kind: str) -> np.ndarray:
    """One period's **fixed within-period waveform** (length ``spc`` = samples-per-cycle),
    standardized to unit scale with a non-negative floor where the archetype demands it.
    This is the shape that *repeats* — the learnable backbone."""
    phase = np.linspace(0.0, 1.0, int(max(2, spc)), endpoint=False)
    if kind == "pulse":                                   # solar: zero night, daytime bell
        start = rng.uniform(0.20, 0.32); width = rng.uniform(0.38, 0.55)
        local = (phase - start) / width
        bell = np.sin(np.pi * np.clip(local, 0, 1)) ** rng.uniform(1.0, 1.8)
        prof = np.where((local >= 0) & (local <= 1), bell, 0.0)
    elif kind == "business":                              # bizitobs: low night, day plateau
        up = 1.0 / (1.0 + np.exp(-(phase - rng.uniform(0.24, 0.34)) / 0.03))
        down = 1.0 / (1.0 + np.exp((phase - rng.uniform(0.80, 0.92)) / 0.03))
        plateau = up * down
        bumps = 1.0 + 0.18 * np.sin(2 * np.pi * rng.uniform(2, 4) * phase + rng.uniform(0, 6))
        prof = 0.06 + plateau * np.clip(bumps, 0.5, None)
    elif kind == "double_hump":                           # electricity: morning + evening peaks
        a1, a2 = rng.uniform(0.5, 0.9), rng.uniform(0.8, 1.2)
        prof = (0.35 + a1 * _raised_bump(phase, rng.uniform(0.30, 0.40), 0.06)
                + a2 * _raised_bump(phase, rng.uniform(0.72, 0.82), 0.07))
    else:                                                 # single_hump: one daytime hump
        prof = 0.15 + _raised_bump(phase, rng.uniform(0.4, 0.6), rng.uniform(0.10, 0.18))
    return prof


def _weekly_factor(rng, n_days: int, *, weekend_low=(0.5, 0.85)) -> np.ndarray:
    """Per-day multiplier with a weekday/weekend contrast (the weekly modulation that
    makes weekly seasonal-naive meaningful). Weekday≈1, weekend lower."""
    wk = np.ones(7)
    wk[5:] = rng.uniform(*weekend_low, size=2)
    wk *= rng.uniform(0.9, 1.1, size=7)                   # mild day-of-week character
    return wk[np.arange(n_days) % 7]


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
    trend: float = 0.0,
) -> Tuple[np.ndarray, int, str]:
    """A **recurring daily profile**: a fixed ``daily_profile`` repeated every ``spc``
    samples, scaled per day by an **autocorrelated** amplitude (``amp_persist`` →
    cross-day persistence + *different heights*, same shape), modulated weekly, plus a
    tiny smooth residual and an optional slow trend.

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
    series = np.concatenate([day_amp[d] * profile for d in range(n_days)])[:n]
    if trend:                                             # optional slow drift of the level
        series = series + trend * np.linspace(0, 1, n)
    resid = noise_amp * S._standardize(_smooth_resid(rng, n))
    out = series + resid
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
    half = int(min(max(1, round(3 * corr)), max(1, n - 1)))
    t = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (t / corr) ** 2); k /= k.sum()
    return np.convolve(rng.normal(0, 1, n), k, mode="same")


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


def samples_per_cycle(period_minutes: float, interval_minutes: float) -> int:
    """samples-per-cycle = period / sampling-interval (the period×frequency decoupling):
    a 1-day cycle (1440 min) is 144 samples at 10-min, 24 at hourly."""
    return int(max(2, round(period_minutes / max(1e-6, interval_minutes))))
