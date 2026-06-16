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

# Features that most distinguish dynamics — rejection must satisfy these.
_KEY_FEATURES = ("seasonal_strength", "trend_strength", "spectral_entropy",
                 "acf1", "intermittency")
_KEY_IDX = tuple(F.FEATURE_NAMES.index(n) for n in _KEY_FEATURES)


def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def _seasonal_component(rng, t: np.ndarray, m: int, seas_str: float) -> np.ndarray:
    """Realistic seasonal shape at period ``m``: a multi-harmonic Fourier series, or
    (for strongly seasonal groups) a sharp **von Mises** peak train — real GIFT-Eval
    traffic/energy series have sharp, multi-harmonic daily cycles that a single
    sinusoid can't reproduce (the spectral_entropy/acf gap in the Tier-1 punch-list)."""
    phase = rng.uniform(0, 2 * np.pi)
    w = 2 * np.pi * t / m
    if seas_str > 0.4 and rng.random() < 0.6:
        # von Mises bump train: sharp periodic peaks (higher κ ⇒ sharper)
        kappa = float(rng.uniform(2.0, 8.0))
        x = np.exp(kappa * (np.cos(w + phase) - 1.0))
    else:
        # Fourier series with 2–4 harmonics and decaying amplitudes
        n_harm = int(rng.integers(2, 5))
        x = np.zeros_like(t)
        for h in range(1, n_harm + 1):
            x += (rng.uniform(0.4, 1.0) / h) * np.sin(h * w + rng.uniform(0, 2 * np.pi))
    return S._standardize(x)


def _impulsive_channel(rng, n: int, m: int, kurt: float) -> np.ndarray:
    """Flat baseline + periodic sharp spikes (at the calendar period and a sub-period)
    + rare heavy-tail outliers — the server/traffic-trace pattern (high excess_kurtosis,
    low seasonal-variance) that smooth seasonal synthesis misses (bitbrains_rnd/5T etc.).
    """
    x = rng.normal(0, 0.15, n)                       # near-flat baseline
    periods = [m] if m and m >= 2 else []
    if m and m // 6 >= 2 and rng.random() < 0.7:     # add a sub-period spike train
        periods.append(int(m // rng.integers(4, 8)))
    for p in periods:
        amp = rng.uniform(2, 6)
        for c in range(p, n, p):                     # spike once per period (jittered)
            j = c + int(rng.integers(-p // 10 - 1, p // 10 + 1))
            if 0 <= j < n:
                x[j] += amp * rng.uniform(0.6, 1.4)
    n_out = int(max(0, rng.normal(2 + 4 * kurt, 1)))  # rare huge outliers
    for _ in range(n_out):
        x[int(rng.integers(n))] += rng.uniform(6, 18) * rng.choice([-1.0, 1.0])
    return S._standardize(x)


def _targeted_channel(rng, n: int, m: int, center: dict) -> np.ndarray:
    """One channel guided by a group's central feature vector ``center`` (a
    name→value dict). Allocates variance to seasonal / trend / noise components so
    the feature extractor recovers the targeted strengths; rejection sampling refines.
    """
    t = np.arange(n, dtype=np.float64)
    trend_str = _clip(center.get("trend_strength", 0.2), 0, 1)
    seas_str = _clip(center.get("seasonal_strength", 0.2), 0, 1)
    acf1 = _clip(center.get("acf1", 0.3), -0.95, 0.95)
    interm = _clip(center.get("intermittency", 0.0), 0, 1)
    kurt = _clip(center.get("excess_kurtosis", 0.0), -1, 1)
    scale = float(np.expm1(_clip(center.get("log_scale", 1.0), 0, 12))) + 1e-3

    # impulsive/spiky regime (heavy-tailed, low seasonal variance): flat baseline +
    # periodic sharp spikes + outliers, not the smooth seasonal+trend+noise mix.
    if kurt > 0.4 and seas_str < 0.3:
        return scale * _impulsive_channel(rng, n, m, kurt) + rng.uniform(-50, 50)

    seasonal = (_seasonal_component(rng, t, m, seas_str)
                if m and m >= 2 and n > 2 * m else np.zeros(n))
    lin = S._standardize(t) if n > 1 else np.zeros(n)
    # heavy-tailed innovations when the group has fat-tailed diffs (Student-t echo)
    if kurt > 0.1:
        df = float(np.clip(30 * (1 - kurt), 3, 30))
        white = S._standardize(rng.standard_t(df, n))
    else:
        white = rng.normal(0, 1, n)
    # RBF-smoothed noise: real test series are smooth (high acf1, low spectral
    # entropy), so a Gaussian-smoothed process matches their spectral shape far better
    # than AR(1)/white. Map the target acf1 to an RBF lengthscale (higher acf1 ⇒
    # longer ℓ ⇒ smoother), which jointly fixes acf1 / stationarity / spectral_entropy.
    ell = float(np.clip(1.0 / max(1e-3, 1.0 - _clip(acf1, 0.0, 0.98)), 1.0, n / 4))
    half = int(min(max(1, 3 * ell), max(1, (n - 1) // 2)))  # keep kernel <= n (conv 'same')
    kt = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (kt / ell) ** 2); kern /= kern.sum()
    noise = S._standardize(np.convolve(white, kern, mode="same"))

    # variance allocation: seasonal -> seas_str, trend -> trend_str·(1-seas_str), rest noise.
    # sqrt weights make the variance *fractions* match (components ~orthogonal, unit var).
    f_s = seas_str
    f_t = trend_str * (1.0 - seas_str)
    f_n = max(0.02, 1.0 - f_s - f_t)
    sig = (np.sqrt(f_s) * seasonal + np.sqrt(f_t) * lin + np.sqrt(f_n) * noise)
    sig = S._standardize(sig)
    if interm > 0.05:  # zero out a fraction to mimic intermittent/sparse series
        sig = sig * (rng.random(n) >= interm)
    return scale * sig + rng.uniform(-50, 50)


def gen_targeted(rng, profile: TestProfile, *, group: Optional[str] = None,
                 n_tries: int = 10) -> Tuple[np.ndarray, int, int, int, str]:
    """Generate one ``synth_targeted`` series matching ``profile``.

    Returns ``(data[C,t], nf, nt, season_length, group)``. Draws a frequency group,
    samples ``(length, season m, C)`` from its marginals, generates a channel guided
    by the group's central features, and keeps the candidate whose feature vector is
    closest to the group center (accepting early once the key features land in band).
    """
    group = group or profile.sample_group(rng)
    center_vec = profile.feature_center(group)
    center = dict(zip(F.FEATURE_NAMES, center_vec))
    scale = profile.feature_scale(group)
    lo, hi = profile.feature_bands(group)

    m = profile.sample_season(group, rng)
    C = profile.sample_n_channels(group, rng)
    # sample length across the group's empirical log_length spread (q05..q95), not a
    # single center value, so the synthetic length distribution matches the test one.
    q = profile.groups[group]["feature_quantiles"]["log_length"]
    n = int(np.clip(round(float(np.expm1(rng.uniform(q[0], q[4])))), 96, 4096))

    log_scale_target = _clip(center.get("log_scale", 1.0), 0, 14)
    best, best_d = None, np.inf
    for _ in range(max(1, n_tries)):
        # Mixture for joint-manifold *diversity* (a single rigid family is trivially
        # separable even when its marginals match): ~40% a profile-rescaled draw from
        # the diverse general families, else the parametric profile-guided synthesis.
        if rng.random() < 0.4:
            data = _general_rescaled(rng, n, C, log_scale_target)
        elif C == 1:
            data = _targeted_channel(rng, n, m, center)[None, :]
        else:  # multivariate: shared seasonal/trend backbone + per-channel variation
            backbone = _targeted_channel(rng, n, m, center)
            data = np.stack([backbone * rng.uniform(0.6, 1.4)
                             + rng.uniform(0.1, 0.4) * np.nanstd(backbone) * rng.normal(0, 1, n)
                             for _ in range(C)])
        feats = F.channel_features(data).mean(axis=0)
        in_band = np.all((feats[list(_KEY_IDX)] >= lo[list(_KEY_IDX)])
                         & (feats[list(_KEY_IDX)] <= hi[list(_KEY_IDX)]))
        d = float(np.mean(((feats - center_vec) / scale) ** 2))
        if d < best_d:
            best, best_d = data, d
        if in_band:
            break
    return best.astype(np.float64), 0, best.shape[0], (m if m and m >= 2 else -1), group


def _general_rescaled(rng, n: int, C: int, log_scale_target: float) -> np.ndarray:
    """A draw from the diverse general families, rescaled to the group's log-scale —
    inherits the general mixture's manifold coverage while staying profile-targeted."""
    from . import synthetic_v2 as V
    names, p = V.general_picker(None)
    fam = names[int(rng.choice(len(names), p=p))]
    if fam == "noise_robust":  # targeted family is not fixed-window; pick a plain one
        fam = "kernelsynth"
    data, _nf, _nt, _m, _kind, _cc, _cp = V.gen_general_series(rng, n, fam)
    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    cur = float(np.nanstd(data)) + 1e-8
    target = float(np.expm1(log_scale_target)) + 1e-3
    return data * (target / cur)


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
