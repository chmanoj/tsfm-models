"""Canonical per-series feature battery (H1) — the *one* feature space shared by
the GIFT-Eval :class:`~tetris.data.test_profile.TestProfile`, the targeted
generator, and the Tier-1 quality harness (C2ST/KS/MMD).

Building all three on the same features is the point: "targeted synth matches the
test data distribution" becomes the *measurable* statement "the C2ST cannot tell
targeted from test on these features", and the targeted generator's rejection step
optimizes against exactly the metric we score it with.

The battery is **hand-rolled, numpy-only** (no catch22/tsfresh/scipy dep), chosen
to span the axes GIFT-Eval actually varies on: scale, length, autocorrelation,
spectral shape, trend/seasonal strength, intermittency, and tail weight. Every
feature is finite by construction (degenerate inputs — constant, all-NaN, length-1
— map to sane defaults), honoring the project rule that nothing downstream ever
sees a non-finite value.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

# The fixed feature order. Anything reading a profile/feature matrix relies on this
# being stable, so append-only if it ever changes.
FEATURE_NAMES: tuple[str, ...] = (
    "log_scale",          # log1p(robust std) — magnitude
    "log_length",         # log1p(t) — series length
    "acf1",               # lag-1 autocorrelation of the level
    "acf_diff1",          # lag-1 autocorrelation of first differences
    "spectral_entropy",   # normalized Shannon entropy of the periodogram in [0,1]
    "dominant_period_frac",  # dominant FFT period / length, in [0,1]
    "trend_strength",     # 1 - var(resid)/var(level) after linear detrend, in [0,1]
    "seasonal_strength",  # seasonal-subseries strength at the dominant period, [0,1]
    "intermittency",      # fraction of near-zero steps, in [0,1]
    "excess_kurtosis",    # tanh-squashed excess kurtosis of diffs (tail/jumps)
    "stationarity",       # var(diff)/var(level) ratio squashed, low => trending/RW
)
N_FEATURES = len(FEATURE_NAMES)

# The *dynamics* (pattern) features — everything except the superficial extent/amplitude
# axes (length, scale). The synth-quality verdict is judged on these so we cannot game
# the C2ST by matching series length/scale (which teach the model no test *pattern*).
# See the [[dont-game-synth-quality-metric]] principle.
_SUPERFICIAL = ("log_scale", "log_length")
DYNAMICS_FEATURES = tuple(n for n in FEATURE_NAMES if n not in _SUPERFICIAL)
DYNAMICS_IDX = tuple(FEATURE_NAMES.index(n) for n in DYNAMICS_FEATURES)

_EPS = 1e-8


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Mean-fill non-finite entries so spectral/ACF estimates stay defined."""
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    if not m.any():
        return np.zeros_like(x)
    if m.all():
        return x
    out = x.copy()
    out[~m] = x[m].mean()
    return out


def _acf1(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < _EPS:
        return 0.0
    return float(np.dot(x[1:], x[:-1]) / denom)


def _spectral(x: np.ndarray) -> tuple[float, float]:
    """(spectral_entropy in [0,1], dominant_period_frac in [0,1])."""
    n = x.size
    if n < 4:
        return 1.0, 0.0
    x = x - x.mean()
    if float(np.dot(x, x)) < _EPS:
        return 0.0, 0.0  # constant -> all power at DC, no spectral spread
    power = np.abs(np.fft.rfft(x)) ** 2
    power = power[1:]  # drop DC
    if power.sum() < _EPS or power.size == 0:
        return 1.0, 0.0
    p = power / power.sum()
    ent = -np.sum(p * np.log(p + _EPS)) / np.log(len(p) + _EPS)
    k = int(np.argmax(power)) + 1  # +1 because DC dropped; freq index
    period = n / k
    return float(np.clip(ent, 0.0, 1.0)), float(np.clip(period / n, 0.0, 1.0))


def _trend_strength(x: np.ndarray) -> float:
    n = x.size
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    var_x = float(x.var())
    if var_x < _EPS:
        return 0.0
    # closed-form linear detrend (avoids lstsq's ill-conditioned-matmul warnings)
    tc = t - t.mean()
    var_t = float(np.dot(tc, tc))
    if var_t < _EPS:
        return 0.0
    slope = float(np.dot(tc, x - x.mean()) / var_t)
    resid = x - (x.mean() + slope * tc)
    return float(np.clip(1.0 - resid.var() / (var_x + _EPS), 0.0, 1.0))


def _seasonal_strength(x: np.ndarray, period: int) -> float:
    n = x.size
    if period < 2 or period > n // 2 or n < 2 * period:
        return 0.0
    var_x = float(x.var())
    if var_x < _EPS:
        return 0.0
    # seasonal-subseries means (phase-averaged), tiled back to length n
    usable = (n // period) * period
    xs = x[:usable].reshape(-1, period)
    phase_mean = xs.mean(axis=0)
    seasonal = np.tile(phase_mean, usable // period)
    resid = x[:usable] - seasonal
    return float(np.clip(1.0 - resid.var() / (var_x + _EPS), 0.0, 1.0))


def _excess_kurtosis(d: np.ndarray) -> float:
    if d.size < 4:
        return 0.0
    d = d - d.mean()
    s2 = float(d.var())
    if s2 < _EPS:
        return 0.0
    k = float((d ** 4).mean() / (s2 ** 2)) - 3.0
    return float(np.tanh(k / 10.0))  # squash heavy-tail outliers into (-1, 1)


def series_features(x: Sequence[float], season: int = 0) -> np.ndarray:
    """Feature vector (length :data:`N_FEATURES`, all finite) for one 1-D series.

    ``season`` (the dataset's calendar period, when known — e.g. 24 hourly, 288 for
    5-min) makes ``seasonal_strength`` measure the strength at *that* period rather
    than the FFT-dominant one. This matters: many real series (server/traffic traces)
    are dominated by a slow drift, so the FFT-dominant period is the trend and the
    real daily seasonality is invisible to a period-agnostic estimate — the strength
    is then taken at the calendar period (max of the two, so a strong detected period
    is never lost)."""
    raw = np.asarray(x, dtype=np.float64).ravel()
    fin = _finite(raw)
    n = raw.size
    if fin.size == 0:
        return np.zeros(N_FEATURES, dtype=np.float32)

    std = float(fin.std())
    log_scale = float(np.log1p(std))
    log_length = float(np.log1p(n))

    filled = _fill_nan(raw)
    acf1 = _acf1(filled)
    diff1 = np.diff(filled)
    acf_diff1 = _acf1(diff1) if diff1.size >= 2 else 0.0

    spec_ent, dom_frac = _spectral(filled)
    period = int(round(n * dom_frac)) if dom_frac > 0 else 0
    trend = _trend_strength(filled)
    seasonal = _seasonal_strength(filled, period)
    if season and season >= 2:  # also measure at the known calendar period
        seasonal = max(seasonal, _seasonal_strength(filled, int(season)))

    # intermittency: fraction of |x| below a small fraction of the series scale
    thr = 0.05 * (np.abs(fin).mean() + _EPS)
    intermittency = float(np.mean(np.abs(fin) <= thr))

    kurt = _excess_kurtosis(diff1) if diff1.size >= 4 else 0.0

    var_x = float(filled.var())
    var_d = float(diff1.var()) if diff1.size >= 2 else var_x
    stationarity = float(np.tanh((var_d / (var_x + _EPS))))  # ~0 trending/RW, ~1 noisy

    out = np.array([
        log_scale, log_length, acf1, acf_diff1, spec_ent, dom_frac,
        trend, seasonal, intermittency, kurt, stationarity,
    ], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def channel_features(data: np.ndarray, season: int = 0) -> np.ndarray:
    """One feature row per channel of a ``[C, t]`` array → ``[C, N_FEATURES]``.
    ``season`` is the dataset calendar period (see :func:`series_features`)."""
    arr = np.atleast_2d(np.asarray(data))
    return np.stack([series_features(arr[c], season=season) for c in range(arr.shape[0])])


def feature_matrix(series: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-series (1-D) feature vectors → ``[N, N_FEATURES]``."""
    if not series:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.stack([series_features(s) for s in series])


def feature_names() -> List[str]:
    return list(FEATURE_NAMES)
