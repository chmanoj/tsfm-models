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

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

from . import features as F
from . import synthetic as S
from .test_profile import TestProfile

def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))


# --- per-config generation knobs (the H1.1 smoothness levers) ----------------
# These are the *fittable* generator parameters (coordinate descent in
# :mod:`tetris.data.synth_fit`). They are derived from the aggregate profile center
# (no leakage) when not yet fitted; the fit refines them per config to match the
# test smoothness/spectral shape. Defaults are deliberately *smooth* (long
# correlation length, tiny white jitter) — the H1 generators drowned every config in
# a white/AR(1) noise floor; the fix is a low-frequency stochastic component plus a
# near-zero high-frequency jitter (see synth_targeted_smoothness_plan.md, lever 1).

@dataclass
class TargetedKnobs:
    corr_len: float = 3.0      # Gaussian smoothing bandwidth (samples) of the smooth-noise
    integrated: bool = False   # integrate the smooth-noise (near-unit-root / trending color)
    white_amp: float = 0.04    # std of the residual *white* jitter, relative to unit signal
    seas_frac: float = 0.6     # variance share of the deterministic seasonal component
    trend_frac: float = 0.2    # variance share of the trend component (structural gen)
    spike_amp: float = 4.0     # cluster-D spike height (in standardized units)
    baseline_amp: float = 0.05  # cluster-D quiet-baseline smooth-noise amplitude

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TargetedKnobs":
        if not d:
            return cls()
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# The smoothness/spectral features the H1.1 work targets are weighted up in the
# candidate-selection + knob-fit objective, so a generator can never trade away
# smoothness to match a superficial feature (the visual gate is the final arbiter).
_SMOOTHNESS_FEATURES = ("acf1", "acf_diff1", "spectral_entropy", "stationarity")


def feature_weights() -> np.ndarray:
    """Per-feature weights for the targeted selection / fit objective: 3× on the
    smoothness/spectral axes, 0 on the superficial extent/amplitude axes (length/scale —
    never matched, per [[dont-game-synth-quality-metric]]), 1 elsewhere."""
    w = np.ones(F.N_FEATURES, dtype=np.float64)
    for name in _SMOOTHNESS_FEATURES:
        w[F.FEATURE_NAMES.index(name)] = 3.0
    for name in ("log_scale", "log_length"):
        w[F.FEATURE_NAMES.index(name)] = 0.0
    return w


def _corr_len_from_acf1(acf1: float) -> float:
    """Gaussian smoothing bandwidth σ that yields lag-1 autocorrelation ``acf1``.
    For a Gaussian kernel the smoothed series has ρ(k)=exp(−k²/(4σ²)), so
    ρ(1)=exp(−1/(4σ²)) ⇒ σ=0.5/√(−ln ρ(1)). Higher acf1 ⇒ longer correlation ⇒ smoother."""
    a = _clip(acf1, 0.30, 0.998)
    return float(np.clip(0.5 / np.sqrt(-np.log(a)), 0.8, 40.0))


def seed_knobs(center: dict) -> TargetedKnobs:
    """Derive a sensible *seed* knob-set from a group's central feature vector — the
    starting point coordinate descent refines. Only aggregate statistics are read
    (acf1/stationarity/seasonal/trend); never a raw test value (no leakage)."""
    acf1 = _clip(center.get("acf1", 0.5), -1, 1)
    stat = _clip(center.get("stationarity", 0.3), 0, 1)
    seas = _clip(center.get("seasonal_strength", 0.3), 0, 0.95)
    trend = _clip(center.get("trend_strength", 0.2), 0, 0.95)
    return TargetedKnobs(
        corr_len=_corr_len_from_acf1(acf1),
        # near-unit-root / strongly-trending configs need an integrated (random-walk-like)
        # smooth-noise color so stationarity (var(diff)/var) gets as low as real.
        integrated=bool(acf1 > 0.95 and stat < 0.08),
        # the white jitter is what made the old synth rough; allow only as much as the
        # config's stationarity (var(diff)/var) actually warrants — near-zero for smooth
        # configs, up to a genuinely-rough level for noisy count/intermittent ones.
        white_amp=float(np.clip(0.8 * stat, 0.01, 0.7)),
        seas_frac=float(seas),
        trend_frac=float(trend * (1 - seas)),
    )


def _smooth_noise(rng, n: int, corr_len: float, *, integrated: bool = False) -> np.ndarray:
    """Low-frequency stochastic component: Gaussian-smoothed white noise (bandwidth
    ``corr_len`` samples), optionally integrated for a near-unit-root color. High
    ``corr_len`` ⇒ high acf1, low spectral-entropy, low stationarity — i.e. *smooth*,
    the way real GIFT-Eval residual variation looks (not white/AR(1) jitter)."""
    if n < 2:
        return rng.normal(0, 1, n)
    sigma = float(max(0.6, corr_len))
    half = int(min(max(1, round(3 * sigma)), max(1, n - 1)))
    t = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    base = np.convolve(rng.normal(0, 1, n), k, mode="same")
    if integrated:
        base = np.cumsum(base - base.mean())   # random-walk-like: acf1→1, stationarity→0
    return S._standardize(base)


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
    periodicity is **regular and forecastable**, not random.

    **Out-of-window long cycles** (lever 5): a period *longer* than the window is **not
    dropped** — its sinusoid over ``n`` points is less than half a cycle, i.e. a *smooth
    partial-cycle drift*, exactly how a long real-world cycle looks in a short window
    (e.g. bizitobs_l2c/5T period 2036, solar/10T period 144 in a 200-pt window). Only
    sub-2-sample periods (aliasing) are skipped."""
    t = np.arange(n, dtype=np.float64)
    x = np.zeros(n)
    for period, weight in (periods or []):
        if period < 2:                                  # sub-sample → aliasing, skip
            continue
        amp = float(np.sqrt(max(weight, 1e-3)))
        phase = rng.uniform(0, 2 * np.pi)
        x += amp * np.sin(2 * np.pi * t / period + phase)
        if sharp and period <= n / 2:                   # harmonics only for in-window cycles
            for h in (2, 3):
                if period / h >= 2:
                    x += (amp / h) * np.sin(2 * np.pi * h * t / period + phase)
    return S._standardize(x) if np.any(x) else x


def _fallback_periods(rng, n: int, m: int, profile, group) -> list:
    periods = profile.dominant_periods(group)
    if not periods and m and m >= 2:
        periods = [[int(m), 1.0]]
    return periods


def _gen_spectral(rng, n, group, profile, center, m, knobs: "TargetedKnobs") -> np.ndarray:
    """PSD / dominant-component synthesis: regular multi-harmonic periodicity from the
    config's dominant periods, blended with a **smooth** low-frequency stochastic
    component (not white/AR jitter) + a tiny white jitter. ``knobs`` set the smoothness
    (correlation length, white amplitude) and the seasonal variance share."""
    seas = _periodic_signal(rng, n, _fallback_periods(rng, n, m, profile, group), sharp=True)
    smooth = _smooth_noise(rng, n, knobs.corr_len, integrated=knobs.integrated)
    if not np.any(seas):
        return S._standardize(smooth + knobs.white_amp * rng.normal(0, 1, n))
    f_s = _clip(knobs.seas_frac, 0.0, 0.97)
    sig = np.sqrt(f_s) * seas + np.sqrt(1 - f_s) * smooth
    return S._standardize(sig + knobs.white_amp * rng.normal(0, 1, n))


def _gen_structural_ar(rng, n, group, profile, center, m, knobs: "TargetedKnobs") -> np.ndarray:
    """level + (smooth) trend (+ occasional level-shift) + seasonal(dominant) + a
    **smooth** low-frequency stochastic component + tiny white jitter — reproduces
    trends/level-shifts/autocorrelation without a high-frequency noise floor."""
    t = np.arange(n, dtype=np.float64)
    seas = _periodic_signal(rng, n, _fallback_periods(rng, n, m, profile, group), sharp=False)
    lin = S._standardize(t) if n > 1 else np.zeros(n)
    if rng.random() < 0.3 and n > 10:                       # occasional level shift
        cut = int(rng.uniform(0.3, 0.7) * n)
        lin = lin.copy(); lin[cut:] += rng.uniform(-2.5, 2.5)
        lin = S._standardize(lin)
    smooth = _smooth_noise(rng, n, knobs.corr_len, integrated=knobs.integrated)
    f_s = _clip(knobs.seas_frac, 0.0, 0.95)
    f_t = _clip(knobs.trend_frac, 0.0, 1.0 - f_s)
    f_n = max(0.03, 1 - f_s - f_t)
    sig = np.sqrt(f_s) * seas + np.sqrt(f_t) * lin + np.sqrt(f_n) * smooth
    return S._standardize(sig + knobs.white_amp * rng.normal(0, 1, n))


def _gen_regular_spikes(rng, n, group, profile, center, m, knobs: "TargetedKnobs" = None) -> np.ndarray:
    """**Genuinely quiet** baseline + **regular**, *sparse* spike train at the dominant
    period: fixed period, fixed phase, near-constant amplitude, minimal jitter → the
    spikes are **predictable**. Real impulsive traces (bitbrains) are *sparse* — a few
    spikes per window on a near-silent baseline — so we enforce a minimum spacing
    (≤ ~12 spikes/window) and widen each spike slightly (a 3-tap raised cosine) rather
    than firing a single sample at every short period."""
    if knobs is None:
        knobs = TargetedKnobs()
    cands = [int(p) for p, _ in (profile.dominant_periods(group) or []) if 2 <= p <= n / 2]
    # prefer a period that yields a *sparse* train (real spikes are rare), floored so we
    # never carpet the window with spikes.
    min_spacing = max(2, n // 12)
    period = next((p for p in sorted(cands, reverse=True) if p >= min_spacing),
                  max(min_spacing, int(m) if m and m >= 2 else n // 8))
    x = knobs.baseline_amp * _smooth_noise(rng, n, max(2.0, knobs.corr_len))  # quiet baseline
    amp = max(2.5, float(knobs.spike_amp))
    phase = int(rng.integers(0, period))                    # consistent offset
    jit = max(0, period // 40)                              # minimal jitter
    for c in range(phase, n, period):
        j = c + (int(rng.integers(-jit, jit + 1)) if jit > 0 else 0)
        h = amp * rng.uniform(0.9, 1.1)                     # near-constant height
        for off, w in ((-1, 0.5), (0, 1.0), (1, 0.5)):     # slightly-widened (3-tap) spike
            if 0 <= j + off < n:
                x[j + off] += w * h
    return S._standardize(x)


def _gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(max(0.6, sigma))
    half = int(min(max(1, round(3 * sigma)), max(1, len(x) - 1)))
    t = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (t / sigma) ** 2); k /= k.sum()
    return np.convolve(x, k, mode="same")


def _smooth_drift(rng, n: int, n_knots: int = 5) -> np.ndarray:
    """A smooth random drift: a few random knots, linearly interpolated then
    Gaussian-smoothed → a slow wandering envelope with no high-frequency content."""
    if n < 4:
        return np.zeros(n)
    knots_t = np.linspace(0, n - 1, max(2, n_knots))
    knots_v = rng.normal(0, 1, len(knots_t)).cumsum()
    drift = np.interp(np.arange(n), knots_t, knots_v)
    return S._standardize(_gauss_smooth(drift, max(2.0, n / (4 * n_knots))))


def _gen_smooth_envelope(rng, n, group, profile, center, m, knobs: "TargetedKnobs") -> np.ndarray:
    """Smooth **deterministic envelope** generators the H1 set lacked (lever 4): logistic /
    Gompertz monotone growth (covid_deaths), Gaussian / raised-cosine bumps
    (bizitobs_service), or a smooth spline drift — plus a tiny smooth-noise residual. For
    the ultra-smooth / near-unit-root configs whose real shape is a clean envelope, not a
    cycle. Predictable by construction (deterministic backbone)."""
    t = np.arange(n, dtype=np.float64)
    kind = rng.integers(0, 4)
    if kind == 0:                                            # logistic growth
        t0 = rng.uniform(0.25, 0.75) * n
        kk = rng.uniform(4.0, 12.0) / n
        env = 1.0 / (1.0 + np.exp(-kk * (t - t0)))
    elif kind == 1:                                          # Gompertz growth
        b = rng.uniform(3.0, 8.0); c = rng.uniform(3.0, 8.0) / n
        env = np.exp(-b * np.exp(-c * t))
    elif kind == 2:                                          # a few smooth bumps
        env = np.zeros(n)
        for _ in range(int(rng.integers(1, 4))):
            c0 = rng.uniform(0, n); w = rng.uniform(0.05, 0.2) * n
            env += rng.uniform(0.5, 1.5) * np.exp(-0.5 * ((t - c0) / w) ** 2)
    else:                                                    # smooth spline drift
        env = _smooth_drift(rng, n, n_knots=int(rng.integers(3, 7)))
    env = S._standardize(env)
    smooth = _smooth_noise(rng, n, max(2.0, knobs.corr_len), integrated=knobs.integrated)
    f_n = _clip(1.0 - max(0.4, knobs.seas_frac + knobs.trend_frac), 0.02, 0.4)
    sig = np.sqrt(1 - f_n) * env + np.sqrt(f_n) * smooth
    return S._standardize(sig + knobs.white_amp * rng.normal(0, 1, n))


_TARGETED_GENERATORS = (_gen_spectral, _gen_structural_ar, _gen_smooth_envelope,
                        _gen_regular_spikes)


def _block_gate(rng, n: int, interm: float) -> np.ndarray:
    """A **soft** 0..1 gate that suppresses the lowest-``interm`` fraction of a smooth
    gate, so off-periods are *contiguous blocks* (solar night, store-closed) with
    *smooth ramps* in/out — not white holes and not hard 0/1 steps. Block intermittency
    with soft edges keeps the active signal smooth; white masking destroyed it."""
    gate = _smooth_noise(rng, n, max(2.0, n / 40))
    thr = float(np.quantile(gate, _clip(interm, 0.0, 0.9)))
    sd = float(np.std(gate)) + 1e-8
    return 1.0 / (1.0 + np.exp(-(gate - thr) / (0.25 * sd)))  # smooth ramp through the threshold


def _shape(rng, n: int, C: int, backbone: np.ndarray, interm: float,
           target_scale: float, corr_len: float = 3.0) -> np.ndarray:
    """Apply intermittency, scale, offset, and (for C>1) per-channel variation. The
    per-channel deviation is **smooth** (low-frequency), not white, so multivariate
    series don't reintroduce the high-frequency noise floor. Intermittency masking is
    applied only when genuinely high (>0.3) — the feature fires spuriously on smooth
    signals that merely cross zero, and masking those would only roughen them."""
    def one(b):
        if interm > 0.3:
            b = b * _block_gate(rng, n, interm)             # soft contiguous off-blocks
        return target_scale * b + rng.uniform(-50, 50)
    if C == 1:
        return one(backbone)[None, :]
    sd = float(np.nanstd(backbone)) + 1e-8
    return np.stack([one(backbone * rng.uniform(0.6, 1.4)
                         + rng.uniform(0.1, 0.4) * sd * _smooth_noise(rng, n, corr_len))
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

    # Per-config smoothness/spectral knobs: the fitted set from the profile if present
    # (synth_fit coordinate descent), else a seed derived from the aggregate center.
    knobs = TargetedKnobs.from_dict(profile.targeted_knobs(group)) \
        if profile.targeted_knobs(group) else seed_knobs(center)

    # Eligible generators gated by config character (so a generator can't be chosen for a
    # config it doesn't suit just because feature-distance happens to favor it):
    #   * regular-spikes only for genuinely impulsive configs — high kurtosis **and a
    #     low-autocorrelation (quiet) baseline** (acf1<0.7); high-acf1 seasonal-but-peaky
    #     load configs (electricity, LOOP) are smooth/periodic, not spike trains;
    #   * smooth-envelope only for weakly-seasonal configs (clean growth/bumps/drift),
    #     never for a strongly-periodic one.
    kurt = _clip(center.get("excess_kurtosis", 0.0), -1, 1)
    seas = _clip(center.get("seasonal_strength", 0.0), 0, 1)
    acf1 = _clip(center.get("acf1", 0.0), -1, 1)
    if kurt > 0.45 and acf1 < 0.7:
        # genuinely impulsive: the picture (quiet baseline + sharp spikes) is unambiguous,
        # so **force** the spike train. Feature-distance alone would wrongly pick a smooth
        # generator here — real impulsive acf1 (~0.5) sits between a spike train's and a
        # smooth signal's, and the smoothness-weighted objective would favour smooth — the
        # exact metric-over-eye trap [[dont-game-synth-quality-metric]] warns against.
        eligible = [_gen_regular_spikes]
    else:
        eligible = [_gen_spectral, _gen_structural_ar]
        if seas < 0.6:
            eligible.append(_gen_smooth_envelope)

    w = feature_weights()
    best, best_d = None, np.inf
    for _ in range(max(1, n_tries)):
        for gen in eligible:                           # goodness-of-fit among eligible
            backbone = gen(rng, n, group, profile, center, m, knobs)
            # the smooth-envelope's low-start already reads as apparent intermittency
            # (growth from ~0); masking it would double-count and chop the growth.
            gi = 0.0 if gen is _gen_smooth_envelope else interm
            data = _shape(rng, n, C, backbone, gi, target_scale, knobs.corr_len)
            feats = F.channel_features(data, season=m).mean(axis=0)
            # smoothness-weighted distance so selection never trades away acf1/spectral
            # shape to match a superficial axis (length/scale carry zero weight).
            d = float(np.sum(w * ((feats - center_vec) / scale_vec) ** 2) / w.sum())
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
