"""GIFT-Eval-targeted synthetic family (H1) — ``synth_targeted``.

Generates series that match the **empirical test data distribution**
(:class:`~tetris.data.test_profile.TestProfile`), not merely the freq/season/horizon
metadata: a frequency group is drawn from the profile, generation knobs are seeded
from that group's central feature vector, and candidates are **rejection-sampled**
against the group's per-feature bands so the output lands in the test feature
manifold. The C2ST harness scores exactly this overlap — targeted should be far
harder to tell from the real test set than the general family is.

No leakage: the profile is aggregate statistics only; nothing here ever reads a real
test value (D13 "match the distribution, never the values").
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import features as F
from . import synthetic as S
from .test_profile import TestProfile

def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def _ar_noise(rng, n: int, ar_coef) -> np.ndarray:
    """AR(p) process with the config's fitted coefficients → the right *predictable*
    autocorrelation (smooth/trending vs mean-reverting). Shrinks explosive roots so the
    process is stationary (predictable, bounded)."""
    coef = [float(c) for c in (ar_coef or [])][:3]
    if not coef or n < 2:
        return rng.normal(0, 1, n)
    s = sum(abs(c) for c in coef)
    if s >= 0.99:
        coef = [c * 0.98 / s for c in coef]
    p = len(coef)
    e = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(n):
        acc = e[i]
        for j in range(p):
            if i - j - 1 >= 0:
                acc += coef[j] * x[i - j - 1]
        x[i] = acc
    return S._standardize(x)


def _periodic_signal(rng, n: int, periods, *, sharp: bool) -> np.ndarray:
    """Sum sinusoids at the config's dominant ``(period, weight)`` components (+ 2nd/3rd
    harmonics when ``sharp`` → peaked shapes). Fixed periods/phase per series ⇒ the
    periodicity is **regular and forecastable**, not random."""
    t = np.arange(n, dtype=np.float64)
    x = np.zeros(n)
    for period, weight in (periods or []):
        if not (2 <= period <= n / 2):
            continue
        amp = float(np.sqrt(max(weight, 1e-3)))
        phase = rng.uniform(0, 2 * np.pi)
        x += amp * np.sin(2 * np.pi * t / period + phase)
        if sharp:
            for h in (2, 3):
                if period / h >= 2:
                    x += (amp / h) * np.sin(2 * np.pi * h * t / period + phase)
    return S._standardize(x) if np.any(x) else x


def _fallback_periods(rng, n: int, m: int, profile, group) -> list:
    periods = profile.dominant_periods(group)
    if not periods and m and m >= 2:
        periods = [[int(m), 1.0]]
    return periods


def _gen_spectral(rng, n, group, profile, center, m) -> np.ndarray:
    """PSD / dominant-component synthesis: regular multi-harmonic periodicity from the
    config's fitted dominant periods + bounded AR noise. Predictable by construction."""
    seas = _periodic_signal(rng, n, _fallback_periods(rng, n, m, profile, group), sharp=True)
    seas_str = _clip(center.get("seasonal_strength", 0.5), 0.1, 0.92)
    noise = _ar_noise(rng, n, profile.ar_coef(group))
    if not np.any(seas):
        return noise
    return S._standardize(np.sqrt(seas_str) * seas + np.sqrt(1 - seas_str) * noise)


def _gen_structural_ar(rng, n, group, profile, center, m) -> np.ndarray:
    """level + trend (+ occasional level-shift) + seasonal(dominant) + AR(p) noise from
    the fitted coefficients — reproduces trends/level-shifts/autocorrelation."""
    t = np.arange(n, dtype=np.float64)
    trend_str = _clip(center.get("trend_strength", 0.2), 0, 1)
    seas_str = _clip(center.get("seasonal_strength", 0.2), 0, 1)
    seas = _periodic_signal(rng, n, _fallback_periods(rng, n, m, profile, group), sharp=False)
    lin = S._standardize(t) if n > 1 else np.zeros(n)
    if rng.random() < 0.3 and n > 10:                       # occasional level shift
        cut = int(rng.uniform(0.3, 0.7) * n)
        lin = lin.copy(); lin[cut:] += rng.uniform(-2.5, 2.5)
        lin = S._standardize(lin)
    noise = _ar_noise(rng, n, profile.ar_coef(group))
    f_s = seas_str; f_t = trend_str * (1 - seas_str); f_n = max(0.05, 1 - f_s - f_t)
    return S._standardize(np.sqrt(f_s) * seas + np.sqrt(f_t) * lin + np.sqrt(f_n) * noise)


def _gen_regular_spikes(rng, n, group, profile, center, m) -> np.ndarray:
    """Flat AR baseline + **regular** spike train(s) at the dominant period(s): fixed
    period, fixed phase, near-constant amplitude, minimal jitter → the spikes are
    **predictable** (the bitbrains pattern done right, not random spikes)."""
    periods = [int(p) for p, _ in (profile.dominant_periods(group) or []) if 2 <= p <= n / 2][:2]
    if not periods:
        periods = [int(m)] if m and 2 <= m <= n / 2 else [max(2, n // 8)]
    x = 0.25 * _ar_noise(rng, n, profile.ar_coef(group))    # quiet baseline
    for period in periods:
        amp = rng.uniform(3.0, 6.0)
        phase = int(rng.integers(0, period))                # consistent offset
        jit = max(0, period // 40)                          # minimal jitter
        for c in range(phase, n, period):
            j = c + (int(rng.integers(-jit, jit + 1)) if jit > 0 else 0)
            if 0 <= j < n:
                x[j] += amp * rng.uniform(0.92, 1.08)       # near-constant height
    return S._standardize(x)


_TARGETED_GENERATORS = (_gen_spectral, _gen_structural_ar, _gen_regular_spikes)


def _shape(rng, n: int, C: int, backbone: np.ndarray, interm: float,
           target_scale: float) -> np.ndarray:
    """Apply intermittency, scale, offset, and (for C>1) per-channel variation."""
    def one(b):
        if interm > 0.05:
            b = b * (rng.random(n) >= interm)
        return target_scale * b + rng.uniform(-50, 50)
    if C == 1:
        return one(backbone)[None, :]
    sd = float(np.nanstd(backbone)) + 1e-8
    return np.stack([one(backbone * rng.uniform(0.6, 1.4)
                         + rng.uniform(0.1, 0.4) * sd * rng.normal(0, 1, n))
                     for _ in range(C)])


def gen_targeted(rng, profile: TestProfile, *, group: Optional[str] = None,
                 n_tries: int = 1) -> Tuple[np.ndarray, int, int, int, str]:
    """Generate one ``synth_targeted`` series matching ``profile``.

    Draws a config group + its ``(length, season, C)`` marginals, then **selects among
    three predictable generators by goodness-of-fit** — spectral/dominant-component,
    structural+AR, and a regular spike-train — keeping the candidate whose feature vector
    is closest to the group center. The chosen generator is whichever best reproduces the
    config (periodic → spectral, trending → structural/AR, impulsive → regular spikes),
    so every series is *learnable* (predictable structure + bounded noise), never random.
    Returns ``(data[C,t], nf, nt, season_length, group)``."""
    group = group or profile.sample_group(rng)
    center_vec = profile.feature_center(group)
    center = dict(zip(F.FEATURE_NAMES, center_vec))
    scale_vec = profile.feature_scale(group)

    m = profile.sample_season(group, rng)
    C = profile.sample_n_channels(group, rng)
    q = profile.groups[group]["feature_quantiles"]["log_length"]
    n = int(np.clip(round(float(np.expm1(rng.uniform(q[0], q[4])))), 96, 4096))
    interm = _clip(center.get("intermittency", 0.0), 0, 1)
    target_scale = float(np.expm1(_clip(center.get("log_scale", 1.0), 0, 14))) + 1e-3

    # Eligible generators gated by config character (so the spike train can't be chosen
    # for a smooth/seasonal config just because feature-distance happens to favor it):
    # regular-spikes only for genuinely impulsive (high-kurtosis) configs.
    kurt = _clip(center.get("excess_kurtosis", 0.0), -1, 1)
    eligible = [_gen_spectral, _gen_structural_ar]
    if kurt > 0.5:
        eligible.append(_gen_regular_spikes)

    best, best_d = None, np.inf
    for _ in range(max(1, n_tries)):
        for gen in eligible:                           # goodness-of-fit among eligible
            backbone = gen(rng, n, group, profile, center, m)
            data = _shape(rng, n, C, backbone, interm, target_scale)
            feats = F.channel_features(data, season=m).mean(axis=0)
            d = float(np.mean(((feats - center_vec) / scale_vec) ** 2))
            if d < best_d:
                best, best_d = data, d
    return best.astype(np.float64), 0, best.shape[0], (m if m and m >= 2 else -1), group


def write_targeted_corpus(
    writer,
    *,
    n_series: int,
    profile: TestProfile,
    seed: int = 0,
    nan_prob: float = 0.15,
    nan_cap: float = 0.15,
    source: str = "synth_targeted",
) -> int:
    """Generate ``n_series`` profile-matched series and feed them to ``writer``.
    Deterministic: series ``idx`` keyed by ``(seed, marker, idx)``."""
    for idx in range(int(n_series)):
        rng = np.random.default_rng((int(seed), 0x7A41, idx))
        data, nf, nt, m, group = gen_targeted(rng, profile)
        if nan_cap > 0 and rng.random() < nan_prob:
            data = S.inject_nans(rng, data, nan_cap)
        data = np.ascontiguousarray(np.atleast_2d(data), dtype=np.float32)
        writer.add(data, nf, nt, season_length=m, source=source,
                   kind=f"targeted_{group}", item_id=f"targeted_{idx}")
    return int(n_series)
