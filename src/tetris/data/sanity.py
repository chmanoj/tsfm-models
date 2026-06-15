"""Simple-synthetic **sanity** stage — periodic series with a *known* season length.

This is the bring-up experiment that precedes any GIFT-Eval training: train on a
small pool of simple periodic series and forecast their held-out horizon, scored
objectively against a **seasonal-naive** baseline (MASE, GIFT-Eval style). The
goal is to prove the architecture has the capacity to learn — first a single sine
wave, then one case at a time, then all cases at once.

The season length ``m`` (weekly=7, monthly=12, daily=24, yearly-on-weekly=52, …)
is **dataset metadata** carried to eval — the model never detects periodicity,
exactly like the GIFT-Eval test split provides the seasonality. One
:class:`SanitySpec` is the single source of truth so the ``sanity`` train loader
and the ``sanity_eval`` eval loader produce the *same* series (train→test on the
same data), with the last ``horizon`` steps held out for scoring.

Generators return raw ``np.float64`` (unnormalized); the loaders cast to float32
and honor the frozen :class:`~tetris.data.contract.Item` contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

SANITY_CASES = (
    "sine_univariate",         # one clean sine wave (the easiest possible signal)
    "multivariate_independent",  # C independent sines, same m (variate-id separation)
    "shared_factor",           # C channels = combos of a shared periodic factor bank (D4)
    "features_target",         # lagged feature drives target (covariate routing, D5/A3)
    "kff_driver",              # target = periodic(past) + unpredictable feature (KFF needed), D11
)


# --- raw periodic generators (known season length m) ------------------------

def _sine(rng, n: int, m: int, noise_frac: float) -> np.ndarray:
    """One noisy sine of period ``m``. Observation noise scales with amplitude so a
    model that learns the underlying sine *beats* seasonal-naive (which propagates
    the noise from one season ago)."""
    t = np.arange(n, dtype=np.float64)
    amp = rng.uniform(1.0, 5.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    offset = rng.uniform(-10.0, 10.0)
    signal = offset + amp * np.sin(2.0 * np.pi * t / m + phase)
    return signal + rng.normal(0.0, noise_frac * amp, n)


def _draw_period(rng, pool) -> int:
    return int(rng.choice(np.asarray(pool)))


def gen_sine_univariate(rng, n: int, C: int, noise_frac: float, pool):
    m = _draw_period(rng, pool)
    return _sine(rng, n, m, noise_frac)[None, :], 0, 1, [m]


def gen_multivariate_independent(rng, n: int, C: int, noise_frac: float, pool):
    """``C`` independent sines, **each with its own frequency drawn per sample** from
    the season pool. No cross-channel signal — the model must learn each channel's
    distinct period via its variate id, and the seasonal-naive baseline is scored
    per channel with that channel's true period."""
    seasons = [_draw_period(rng, pool) for _ in range(C)]
    x = np.stack([_sine(rng, n, seasons[c], noise_frac) for c in range(C)])
    return x, 0, C, seasons


def gen_shared_factor(rng, n: int, C: int, noise_frac: float, pool, *, bank: int = 4):
    """``C`` channels = random linear combos of a shared bank of sine factors whose
    **periods are drawn per sample** from the pool + idiosyncratic noise + per-channel
    scale/offset. Channels correlate through the shared factors (the D4 cross-channel
    routing signal). Each channel is a mixture, so its declared seasonality is the
    dominant factor period (the naive reference)."""
    fper = [_draw_period(rng, pool) for _ in range(bank)]
    factors = np.stack([_sine(rng, n, fper[k], noise_frac=0.0) for k in range(bank)])
    out = np.empty((C, n), dtype=np.float64)
    seasons: list[int] = []
    for c in range(C):
        k = int(rng.integers(1, bank + 1))
        sel = rng.choice(bank, size=k, replace=False)
        w = rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)
        sig = (w[:, None] * factors[sel]).sum(0)
        sig = sig / (sig.std() + 1e-9)
        scale = np.exp(rng.uniform(-1.0, 1.0))
        offset = rng.uniform(-10.0, 10.0)
        out[c] = scale * (sig + noise_frac * rng.normal(0.0, 1.0, n)) + offset
        # declared season = period of this channel's largest-weight factor
        seasons.append(int(fper[sel[int(np.argmax(np.abs(w)))]]))
    return out, 0, C, seasons


def gen_features_target(rng, n: int, C: int, noise_frac: float, pool):
    """One feature (period drawn per sample) drives a target that reads it lagged by
    ``k`` steps + small noise. Features-first ``[feature; target]`` (nf=1, nt=1) —
    a channel-independent baseline cannot capture the lagged edge. Target inherits
    the feature's period."""
    m = _draw_period(rng, pool)
    feature = _sine(rng, n, m, noise_frac)
    lag = int(rng.integers(1, min(m, n // 4) + 1))
    target = np.empty(n, dtype=np.float64)
    target[:lag] = feature[0]
    target[lag:] = feature[:-lag]
    target = target + rng.normal(0.0, noise_frac * feature.std(), n)
    return np.stack([feature, target]), 1, 1, [m, m]


# Fraction of every target step that comes from the contemporaneous known-future
# driver (the rest comes from the target's own past-derivable structure). High, so
# a past-only model structurally cannot do well — it must learn to read KFF. The
# horizon has few query patches, so we weight KFF heavily to force the model to use
# it (per the data-design note: target_t = (1-w)·past + w·kff_t).
_KFF_WEIGHT = 0.7


def gen_kff_driver(rng, n: int, C: int, noise_frac: float, pool):
    """``target_t = (1-w)·periodic_t  +  w·driver_t``  with ``w = _KFF_WEIGHT``.

    Features-first ``[driver; target]`` (nf=1, nt=1). The **driver** is a dense,
    **aperiodic** known-future covariate (standardized random walk) — unpredictable
    from its own past beyond persistence, so a past-only model misses most of every
    horizon step; with the driver's known future revealed (KFF, D11) the target is
    recoverable. The **periodic** part is past-derivable (the model learns ``m`` from
    history), so the model must combine *past* (periodic) and *known-future* (driver)
    information. Because the driver is dense and dominant, the loss rewards using KFF
    at every step (unlike a sparse signal the model can write off)."""
    m = _draw_period(rng, pool)
    t = np.arange(n, dtype=np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    periodic = np.sin(2.0 * np.pi * t / m + phase)                  # ~unit, past-derivable
    driver = np.cumsum(rng.normal(0.0, 1.0, n))                     # aperiodic random walk
    driver = (driver - driver.mean()) / (driver.std() + 1e-9)       # standardized, ~unit
    w = _KFF_WEIGHT
    amp = rng.uniform(1.0, 4.0)
    offset = rng.uniform(-5.0, 5.0)
    target = offset + amp * ((1.0 - w) * periodic + w * driver) + rng.normal(0.0, noise_frac, n)
    return np.stack([driver, target]), 1, 1, [m, m]


_GENERATORS = {
    "sine_univariate": gen_sine_univariate,
    "multivariate_independent": gen_multivariate_independent,
    "shared_factor": gen_shared_factor,
    "features_target": gen_features_target,
    "kff_driver": gen_kff_driver,
}


# --- the matched train/eval source of truth ---------------------------------

@dataclass
class SanitySpec:
    """Generates one reproducible periodic series per index, the single source of
    truth shared by the ``sanity`` train loader and the ``sanity_eval`` shard.

    ``make(idx)`` returns ``(data[C, series_len], nf, nt, m)`` raw; the last
    ``horizon`` steps are the held-out forecast target. ``n_series`` distinct
    series form the overfit pool (the train loader cycles over them)."""

    case: str = "sine_univariate"
    season_lengths: Tuple[int, ...] = (24,)
    series_len: int = 512
    horizon: int = 32
    n_channels: int = 4
    channels_distribution: Tuple[int, ...] = ()
    n_series: int = 64
    noise_frac: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.case not in _GENERATORS:
            raise ValueError(f"unknown sanity case {self.case!r}; pick from {SANITY_CASES}")
        if not self.season_lengths:
            raise ValueError("season_lengths must be non-empty")
        if self.horizon >= self.series_len:
            raise ValueError(f"horizon {self.horizon} must be < series_len {self.series_len}")
        self.season_lengths = tuple(int(m) for m in self.season_lengths)
        self.channels_distribution = tuple(int(c) for c in self.channels_distribution)
        if self.channels_distribution and len(self.channels_distribution) != 2:
            raise ValueError("channels_distribution must be empty or [lo, hi]")

    @classmethod
    def from_cfg(cls, cfg) -> "SanitySpec":
        d = cfg.data
        return cls(
            case=d.case,
            season_lengths=tuple(d.season_lengths),
            series_len=d.series_len,
            horizon=d.horizon,
            n_channels=d.n_channels,
            channels_distribution=tuple(d.channels_distribution),
            n_series=d.n_series,
            seed=cfg.run.seed,
        )

    def channels_of(self, rng) -> int:
        """Per-sample channel count: drawn from ``channels_distribution`` when set
        (multivariate cases vary C across samples), else fixed ``n_channels``. The
        case generator may still override (univariate→1, features_target→2)."""
        if self.channels_distribution:
            lo, hi = self.channels_distribution
            return int(rng.integers(lo, hi + 1))
        return self.n_channels

    def make(self, idx: int):
        """Raw full series for ``idx`` → ``(data[C, series_len], nf, nt, seasons)``.

        Every channel's frequency is drawn per sample from ``season_lengths`` (so a
        given channel index does not have a fixed period across samples). ``seasons``
        is the per-channel period list (length ``C``); the series-level seasonality
        declared to MASE is ``seasons[nf]`` (the first target channel)."""
        rng = np.random.default_rng((self.seed, idx))
        C = self.channels_of(rng)
        data, nf, nt, seasons = _GENERATORS[self.case](
            rng, self.series_len, C, self.noise_frac, self.season_lengths
        )
        return (np.ascontiguousarray(data, dtype=np.float64), int(nf), int(nt),
                [int(s) for s in seasons])

    @property
    def context_len(self) -> int:
        return self.series_len - self.horizon
