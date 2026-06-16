"""Bigger, more varied synthetic corpus (G4) — materialized to disk.

Extends ``data/sanity.py``'s periodic cases into a broad pretrain-style mixture by
composing the ``synthetic.py`` primitives (trend / AR / regime / intermittent /
random-walk / seasonal / shared-factor / lag-probe) and adding two G4 families:

* **multi-seasonal** — superposed sines at several *integer* calendar periods
  (e.g. daily 24 + weekly 168), so a dataset-level ``season_length`` is meaningful
  for MASE where this corpus is ever an eval source.
* **KFF driver (lead-lag covariate)** — feature channel(s) whose *lagged*,
  optionally known-future values drive the target (D11), planting genuine
  cross-channel dependence the channel-independent baseline cannot capture.

Each series is generated deterministically from ``(seed, "synth", idx)`` so the
corpus is reproducible and order-independent. The generator yields raw, possibly
NaN-bearing series plus per-series metadata (``nf``/``nt``, ``season_length``,
``kind``) carried into the shard index.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import synthetic as S

# Integer calendar periods used by the seasonal families (so season_length is a
# real, MASE-usable integer rather than the float periods of S.gen_seasonal).
_CALENDAR_PERIODS = (4, 7, 12, 24, 48, 96, 144, 168, 336, 52)

# Family mixture (renormalized). Univariate primitives dominate; multivariate
# (shared-factor) and lead-lag (KFF driver) carry the cross-channel signal.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "univariate": 0.28,       # one of the 8 raw primitives (trend/AR/regime/...)
    "seasonal_known": 0.14,   # single integer-period seasonal (+ optional trend)
    "multi_seasonal": 0.14,   # superposed integer-period sines
    "intermittent": 0.08,     # Croston-style sparse
    "regime_jump": 0.08,      # level-shift stress (D10)
    "shared_factor": 0.18,    # multivariate latent-factor bank (lead-lag inside)
    "kff_driver": 0.10,       # lead-lag known-future covariate (D11)
}

_PRIMITIVE_SUBSET = ("linear_trend", "exp_trend", "ar", "random_walk", "gbm", "regime_jump")


# --- new G4 generators -------------------------------------------------------

def gen_multi_seasonal(rng, n: int, *, k: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Sum of ``k`` integer-period sines (+ small noise). Returns ``(x, m)`` where
    ``m`` is the dominant (largest-amplitude) period — the dataset seasonality."""
    usable = [p for p in _CALENDAR_PERIODS if p < n / 2] or [min(_CALENDAR_PERIODS)]
    k = int(rng.integers(2, 4)) if k is None else k
    k = min(k, len(usable))
    periods = rng.choice(usable, size=k, replace=False)
    t = np.arange(n, dtype=np.float64)
    x = np.zeros(n)
    amps = []
    for p in periods:
        amp = rng.uniform(1, 10)
        phase = rng.uniform(0, 2 * np.pi)
        x += amp * np.sin(2 * np.pi * t / p + phase)
        amps.append(amp)
    x += rng.normal(0, 1.0, n)
    if rng.random() < 0.5:  # optional trend overlay (exercise D10 log region)
        x += S.gen_linear_trend(rng, n, noise=0.0)
    dominant = int(periods[int(np.argmax(amps))])
    return x, dominant


def gen_seasonal_known(rng, n: int) -> Tuple[np.ndarray, int]:
    """One integer-period seasonal (optionally trended). Returns ``(x, m)``."""
    usable = [p for p in _CALENDAR_PERIODS if p < n / 2] or [min(_CALENDAR_PERIODS)]
    m = int(rng.choice(usable))
    t = np.arange(n, dtype=np.float64)
    amp = rng.uniform(2, 15)
    phase = rng.uniform(0, 2 * np.pi)
    x = amp * np.sin(2 * np.pi * t / m + phase) + rng.normal(0, 1.0, n)
    if rng.random() < 0.4:
        x = x + S.gen_linear_trend(rng, n, noise=0.0)
    return x, m


def gen_kff_driver(rng, n: int, *, max_lag: int = 24) -> Tuple[np.ndarray, int, int, int]:
    """Lead-lag known-future covariate (D11). ``nf`` seasonal feature channels
    drive the (single) target as a weighted sum of their *lagged* values + noise.
    Returns ``(data[C, n], nf, nt, season_length)`` features-first; the dominant
    feature period is reported as the series seasonality."""
    nf = int(rng.integers(1, 3))   # 1-2 driver channels
    usable = [p for p in _CALENDAR_PERIODS if p < n / 2] or [min(_CALENDAR_PERIODS)]
    feats = []
    seasons = []
    target = np.zeros(n)
    for _ in range(nf):
        m = int(rng.choice(usable))
        seasons.append(m)
        tt = np.arange(n, dtype=np.float64)
        f = rng.uniform(2, 10) * np.sin(2 * np.pi * tt / m + rng.uniform(0, 2 * np.pi))
        f = f + rng.normal(0, 0.5, n)
        feats.append(f)
        lag = int(rng.integers(1, max_lag + 1))
        w = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
        target += w * S._shift(f, lag)
    target += rng.normal(0, 0.3, n)
    data = np.stack(feats + [target])           # features-first, nt = 1
    dominant = int(max(seasons))
    return data, nf, 1, dominant


# --- family dispatch ---------------------------------------------------------

def gen_series(rng, n: int, family: str) -> Tuple[np.ndarray, int, int, int, str]:
    """Generate one corpus series. Returns ``(data[C, n], nf, nt, season_length, kind)``;
    ``season_length`` is ``-1`` when the family has no dataset-level period."""
    if family == "univariate":
        kind = _PRIMITIVE_SUBSET[int(rng.integers(len(_PRIMITIVE_SUBSET)))]
        x = S.gen_primitive(rng, n, kind=kind)
        return x[None, :], 0, 1, -1, kind
    if family == "seasonal_known":
        x, m = gen_seasonal_known(rng, n)
        return x[None, :], 0, 1, m, "seasonal_known"
    if family == "multi_seasonal":
        x, m = gen_multi_seasonal(rng, n)
        return x[None, :], 0, 1, m, "multi_seasonal"
    if family == "intermittent":
        x = S.gen_intermittent(rng, n)
        return x[None, :], 0, 1, -1, "intermittent"
    if family == "regime_jump":
        x = S.gen_regime_jump(rng, n)
        return x[None, :], 0, 1, -1, "regime_jump"
    if family == "shared_factor":
        C = int(rng.integers(2, 9))
        x = S.gen_shared_factor(rng, n, C)
        return x, 0, C, -1, "shared_factor"
    if family == "kff_driver":
        data, nf, nt, m = gen_kff_driver(rng, n)
        return data, nf, nt, m, "kff_driver"
    raise ValueError(f"unknown synthetic family {family!r}")


def _family_picker(weights: Dict[str, float]):
    names = list(weights.keys())
    w = np.array([weights[k] for k in names], dtype=np.float64)
    if (w <= 0).all():
        raise ValueError("synthetic family weights must have positive total")
    p = w / w.sum()
    return names, p


def write_synthetic_corpus(
    writer,
    *,
    n_series: int,
    seed: int = 0,
    length_range: Tuple[int, int] = (96, 4096),
    nan_prob: float = 0.25,
    nan_cap: float = 0.2,
    weights: Optional[Dict[str, float]] = None,
    source: str = "synthetic",
) -> int:
    """Generate ``n_series`` varied synthetic series and feed them to ``writer``.

    Deterministic: series ``idx`` is keyed by ``(seed, "synth", idx)`` independent
    of order. Returns the number of series written."""
    names, p = _family_picker(weights or DEFAULT_WEIGHTS)
    lo, hi = int(length_range[0]), int(length_range[1])
    for idx in range(int(n_series)):
        rng = np.random.default_rng((int(seed), 0x5917, idx))
        n = int(rng.integers(lo, hi + 1))
        family = names[int(rng.choice(len(names), p=p))]
        data, nf, nt, m, kind = gen_series(rng, n, family)
        if nan_cap > 0 and rng.random() < nan_prob:
            data = S.inject_nans(rng, data, nan_cap)
        data = np.ascontiguousarray(np.atleast_2d(data), dtype=np.float32)
        writer.add(data, nf, nt, season_length=m, source=source, kind=kind,
                   item_id=f"synth_{idx}")
    return int(n_series)


def family_kinds() -> List[str]:
    return list(DEFAULT_WEIGHTS.keys())
