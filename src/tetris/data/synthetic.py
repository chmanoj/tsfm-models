"""Synthetic generators (§5.1) — raw, unnormalized, reproducible by seed.

Univariate primitives (trend, seasonal, AR, intermittent, random-walk/GBM,
regime jump) exercise the D10 normalization regimes; the shared-factor generator
(Mystic-B echo, D13-C) is the primary cross-channel training signal — channels
correlate through a shared latent factor bank, with optional lead-lag; the
lag-probe (A3 diagnostic, D5) plants a clean lagged feature->target dependence
that a channel-independent baseline must fail to capture.

All generators return ``np.float64`` arrays (raw values); the loader injects
NaNs and casts to float32.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Curated, non-degenerate primitives used as shared latent factors (smooth-ish,
# so correlations through shared factors are stable and measurable).
_FACTOR_KINDS = ("seasonal", "ar", "random_walk", "linear_trend")
PRIMITIVE_KINDS = (
    "linear_trend", "exp_trend", "seasonal", "ar",
    "intermittent", "random_walk", "gbm", "regime_jump",
)


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else x - x.mean()


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    """Causal shift by ``lag`` steps (x[t] <- x[t-lag]); edge-padded with x[0]."""
    if lag <= 0:
        return x.copy()
    out = np.empty_like(x)
    out[:lag] = x[0]
    out[lag:] = x[:-lag]
    return out


# --- univariate primitives --------------------------------------------------

def gen_linear_trend(rng, n, slope=None, noise=1.0):
    t = np.arange(n, dtype=np.float64)
    slope = rng.uniform(-2, 2) if slope is None else slope
    return slope * t + rng.normal(0, noise, n) + rng.uniform(-50, 50)


def gen_exp_trend(rng, n, rate=None, noise=1.0):
    t = np.arange(n, dtype=np.float64)
    # Bound TOTAL growth so long series can't overflow float32 in the downstream
    # variance/normalization (a fixed per-step rate makes exp(rate*t) explode for
    # large n: ~1e29 at n=4096, whose square overflows fp32 -> NaN/divergence).
    # Grow by exp(1)..exp(6) (~3x..400x) end-to-end regardless of length.
    if rate is None:
        rate = rng.uniform(1.0, 6.0) / max(1, n - 1)
    return np.exp(rate * t) + rng.normal(0, noise, n)


def gen_seasonal(rng, n, n_components=None, noise=1.0):
    t = np.arange(n, dtype=np.float64)
    k = rng.integers(1, 4) if n_components is None else n_components
    x = np.zeros(n)
    for _ in range(k):
        period = rng.uniform(4, max(8.0, n / 3))
        amp = rng.uniform(1, 10)
        phase = rng.uniform(0, 2 * np.pi)
        x += amp * np.sin(2 * np.pi * t / period + phase)
    return x + rng.normal(0, noise, n)


def gen_ar(rng, n, p=None, noise=1.0):
    p = int(rng.integers(1, 4)) if p is None else p
    # random coefficients scaled so Σ|a| < 0.95 (sufficient for stability)
    a = rng.uniform(-1, 1, p)
    a *= 0.95 / (np.abs(a).sum() + 1e-9)
    x = np.zeros(n)
    eps = rng.normal(0, noise, n)
    for i in range(n):
        acc = eps[i]
        for j in range(p):
            if i - j - 1 >= 0:
                acc += a[j] * x[i - j - 1]
        x[i] = acc
    return x


def gen_intermittent(rng, n, rate=None):
    rate = rng.uniform(0.05, 0.2) if rate is None else rate
    mask = rng.random(n) < rate
    x = np.zeros(n)
    x[mask] = rng.uniform(1, 10, int(mask.sum()))
    return x


def gen_random_walk(rng, n, drift=None, noise=1.0):
    drift = rng.uniform(-0.1, 0.1) if drift is None else drift
    return np.cumsum(rng.normal(drift, noise, n)) + rng.uniform(-20, 20)


def gen_gbm(rng, n, mu=None, sigma=None):
    mu = rng.uniform(-0.001, 0.002) if mu is None else mu
    sigma = rng.uniform(0.005, 0.03) if sigma is None else sigma
    log_ret = rng.normal(mu, sigma, n)
    return 100.0 * np.exp(np.cumsum(log_ret))


def gen_regime_jump(rng, n, noise=1.0):
    x = gen_random_walk(rng, n, noise=noise)
    cut = int(rng.uniform(0.3, 0.7) * n)
    x[cut:] += rng.uniform(50, 500) * rng.choice([-1, 1])
    return x


_PRIMITIVE_FNS = {
    "linear_trend": gen_linear_trend,
    "exp_trend": gen_exp_trend,
    "seasonal": gen_seasonal,
    "ar": gen_ar,
    "intermittent": gen_intermittent,
    "random_walk": gen_random_walk,
    "gbm": gen_gbm,
    "regime_jump": gen_regime_jump,
}


def gen_primitive(rng, n, kind=None) -> np.ndarray:
    """Generate one univariate primitive (random kind if not specified)."""
    if kind is None:
        kind = PRIMITIVE_KINDS[rng.integers(len(PRIMITIVE_KINDS))]
    return _PRIMITIVE_FNS[kind](rng, n)


# --- multivariate shared-factor (Mystic-B echo, D13-C) ----------------------

def gen_shared_factor(
    rng,
    n,
    C,
    *,
    bank_size=7,
    kernels_per_variate=(3, 4),
    idiosyncratic=0.3,
    lag_prob=0.3,
    max_lag=20,
) -> np.ndarray:
    """``[C, n]`` channels = linear combos of a shared ``bank_size``-factor bank
    (each channel draws 3–4 factors) + idiosyncratic noise + per-channel
    scale/offset; some channels read a *lagged* factor (lead-lag). Channels that
    share factors correlate — the cross-channel routing signal (D4)."""
    factors = np.stack([
        _standardize(_PRIMITIVE_FNS[_FACTOR_KINDS[rng.integers(len(_FACTOR_KINDS))]](rng, n))
        for _ in range(bank_size)
    ])  # [bank_size, n]

    lo, hi = kernels_per_variate
    out = np.empty((C, n), dtype=np.float64)
    for c in range(C):
        k = int(rng.integers(lo, hi + 1))
        sel = rng.choice(bank_size, size=min(k, bank_size), replace=False)
        sig = np.zeros(n)
        for f in sel:
            w = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
            fac = factors[f]
            if rng.random() < lag_prob:
                fac = _shift(fac, int(rng.integers(1, max_lag + 1)))
            sig += w * fac
        sig = _standardize(sig)
        sig = sig + idiosyncratic * rng.normal(0, 1, n)
        scale = np.exp(rng.uniform(-2, 2))     # per-channel scale (D10 receipts)
        offset = rng.uniform(-100, 100)
        out[c] = scale * sig + offset
    return out


# --- lag-probe (A3 diagnostic, D5) ------------------------------------------

def gen_lag_probe(rng, n, *, lag=None, max_lag=20, noise=0.1) -> Tuple[np.ndarray, int]:
    """``([2, n], lag)`` — row 0 a white-noise feature, row 1 the target =
    feature lagged-``k`` + small noise. Features-first. The channel-independent
    baseline cannot capture this; a model that routes the lagged cross-channel
    edge can (used to detect routing failure → A3 ladder)."""
    lag = int(rng.integers(1, max_lag + 1)) if lag is None else lag
    feature = rng.normal(0, 1, n)
    target = _shift(feature, lag) + rng.normal(0, noise, n)
    return np.stack([feature, target]), lag


# --- NaN injection (D7) -----------------------------------------------------

def inject_nans(rng, x: np.ndarray, cap: float, full_patch_prob: float = 0.2) -> np.ndarray:
    """Inject scattered NaNs (fraction ≤ ``cap``) and, with ``full_patch_prob``,
    one fully-missing contiguous window in a single channel (D7 [NA])."""
    if cap <= 0:
        return x
    y = np.array(x, dtype=np.float64, copy=True)
    if y.ndim == 1:
        y = y[None, :]
        squeeze = True
    else:
        squeeze = False
    C, n = y.shape
    frac = rng.uniform(0, cap)
    mask = rng.random((C, n)) < frac
    y[mask] = np.nan
    if rng.random() < full_patch_prob and n >= 8:
        c = int(rng.integers(C))
        w = int(rng.integers(2, max(3, n // 8)))
        s = int(rng.integers(0, n - w))
        y[c, s:s + w] = np.nan
    return y[0] if squeeze else y
