"""Synthetic corpus v2 generators (H1) — the ``synth_general`` family.

Composes (never replaces) the G4 primitives in :mod:`tetris.data.synthetic` into a
broader, **multivariate-first** mixture aimed at teaching general temporal structure
and noise-robustness before the GIFT-Eval-*targeted* family (see
:mod:`tetris.data.synthetic_targeted`) narrows onto the test distribution. Families:

* **kernelsynth** — multivariate GP-*kernel-composition* (KernelSynth / Mystic-B,
  D13-C): a shared bank of factors, each a random ``+``/``×`` composition of base
  kernels (linear / RBF-smooth / periodic / rational-quadratic / white); every
  channel is a random linear combo of the bank. Univariate is just ``C = 1``.
* **sde** — stochastic processes with full observed dynamics: Ornstein–Uhlenbeck
  (mean-reverting), GBM, jump-diffusion, and volatility-clustering (ARCH-ish).
* **noise_robust** — the Path-B construction (the maintainer's idea): a series of
  length ``ctx + horizon`` whose **context is the noisy realization** and whose
  **horizon is the clean conditional mean** ``E[x_{t+h} | F_ctx]``. Carried with a
  fixed-window crop hint so the reservoir crops at exactly ``(ctx, horizon)`` and
  the horizon GT stays clean — training the model to predict the predictable part,
  not the noise (which it cannot forecast). No loss change: scoring the clean GT is
  automatic once the crop lands on the baked boundary.
* **structural_ts** — trend × multi-seasonal × (optional holiday spikes) + noise.
* plus the G4 ``multi_seasonal`` / ``intermittent`` / ``regime_jump`` /
  ``shared_factor`` families.

Independently, **dilution channels** (D5/D13) are injected into any sample with a
configurable probability: extra *feature* channels that are pure noise or weak-signal
distractors, which the model must learn to ignore (targets are never touched).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import synthetic as S

_CALENDAR_PERIODS = (4, 7, 12, 24, 48, 96, 144, 168, 336, 52)

# General-family mixture (renormalized). Multivariate KernelSynth dominates; the
# SDE + noise-robust families carry the stochastic-process / robustness signal.
GENERAL_WEIGHTS: Dict[str, float] = {
    "kernelsynth": 0.30,    # multivariate composed-kernel GP (the core)
    "sde": 0.15,            # OU / GBM / jump-diffusion / vol-clustering (observed)
    "noise_robust": 0.15,   # noisy context + clean conditional-mean horizon (Path B)
    "structural_ts": 0.13,  # trend × multi-season × holiday
    "multi_seasonal": 0.10,
    "shared_factor": 0.07,  # G4 multivariate latent-factor bank
    "regime_jump": 0.05,
    "intermittent": 0.05,
}

_SDE_KINDS = ("ou", "gbm", "jump_diffusion", "vol_cluster")
_NOISE_ROBUST_KINDS = ("ou", "rw_drift", "gbm", "seasonal_ar")


# --- composed-kernel factors (KernelSynth, cheap O(n) realizations) ----------

def _rbf_smooth(rng, n: int) -> np.ndarray:
    """RBF kernel echo: Gaussian-smoothed white noise (a smooth random function)."""
    ell = float(rng.uniform(3, max(6.0, n / 8)))
    half = int(min(max(1, 3 * ell), n))
    t = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (t / ell) ** 2)
    k /= k.sum()
    return np.convolve(rng.normal(0, 1, n), k, mode="same")


def _periodic(rng, n: int) -> np.ndarray:
    usable = [p for p in _CALENDAR_PERIODS if p < n / 2] or [min(_CALENDAR_PERIODS)]
    p = float(rng.choice(usable))
    return np.sin(2 * np.pi * np.arange(n) / p + rng.uniform(0, 2 * np.pi))


def _linear(rng, n: int) -> np.ndarray:
    return rng.uniform(-2, 2) * (np.arange(n, dtype=np.float64) / max(1, n))


def _rational_quadratic(rng, n: int) -> np.ndarray:
    """Mixture of RBF lengthscales (a rational-quadratic kernel echo)."""
    return sum(_rbf_smooth(rng, n) for _ in range(int(rng.integers(2, 4))))


_BASE_KERNELS = (_rbf_smooth, _periodic, _linear, _rational_quadratic,
                 lambda rng, n: rng.normal(0, 1, n))  # white


def _compose_kernel(rng, n: int) -> np.ndarray:
    """One factor = a random ``+``/``×`` composition of 1–3 base kernels."""
    k = int(rng.integers(1, 4))
    sig = None
    for _ in range(k):
        base = _BASE_KERNELS[int(rng.integers(len(_BASE_KERNELS)))](rng, n)
        base = S._standardize(base)
        if sig is None:
            sig = base
        elif rng.random() < 0.5:
            sig = sig + base
        else:
            sig = sig * base
    return S._standardize(np.asarray(sig))


def gen_kernelsynth(rng, n: int, C: int, *, bank_size: int = 6) -> np.ndarray:
    """``[C, n]`` multivariate KernelSynth: channels are random linear combos of a
    shared composed-kernel factor bank (so they correlate through latent factors)."""
    bank = np.stack([_compose_kernel(rng, n) for _ in range(bank_size)])
    out = np.empty((C, n), dtype=np.float64)
    for c in range(C):
        k = int(rng.integers(1, min(4, bank_size) + 1))
        sel = rng.choice(bank_size, size=k, replace=False)
        sig = sum(rng.uniform(-1.5, 1.5) * bank[f] for f in sel)
        sig = S._standardize(sig) + rng.uniform(0.05, 0.4) * rng.normal(0, 1, n)
        out[c] = np.exp(rng.uniform(-2, 2)) * sig + rng.uniform(-50, 50)
    return out


# --- stochastic processes (observed full dynamics) ---------------------------

def gen_ou(rng, n: int, *, theta=None, mu=None, sigma=None, x0=None) -> np.ndarray:
    """Ornstein–Uhlenbeck (mean-reverting): dx = θ(μ−x)dt + σ dW (dt=1)."""
    theta = float(rng.uniform(0.02, 0.3)) if theta is None else theta
    mu = float(rng.uniform(-20, 20)) if mu is None else mu
    sigma = float(rng.uniform(0.5, 5)) if sigma is None else sigma
    x = np.empty(n)
    x[0] = float(rng.uniform(-20, 20)) if x0 is None else x0
    eps = rng.normal(0, 1, n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * eps[t]
    return x


def gen_jump_diffusion(rng, n: int) -> np.ndarray:
    """Brownian drift + rare Poisson jumps (Merton-style)."""
    drift = rng.uniform(-0.05, 0.05)
    sigma = rng.uniform(0.3, 2.0)
    x = np.cumsum(rng.normal(drift, sigma, n))
    rate = rng.uniform(0.005, 0.03)
    jumps = (rng.random(n) < rate) * rng.normal(0, rng.uniform(10, 50), n)
    return x + np.cumsum(jumps)


def gen_vol_cluster(rng, n: int) -> np.ndarray:
    """Volatility clustering (ARCH-ish): variance follows squared past shocks."""
    omega, alpha, beta = 1.0, rng.uniform(0.1, 0.3), rng.uniform(0.5, 0.8)
    var = np.empty(n)
    var[0] = omega / max(1e-3, 1 - alpha - beta)
    eps = np.empty(n)
    eps[0] = np.sqrt(var[0]) * rng.normal()
    for t in range(1, n):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * rng.normal()
    return np.cumsum(eps)


def gen_sde(rng, n: int) -> Tuple[np.ndarray, str]:
    kind = _SDE_KINDS[int(rng.integers(len(_SDE_KINDS)))]
    if kind == "ou":
        return gen_ou(rng, n), kind
    if kind == "gbm":
        return S.gen_gbm(rng, n), kind
    if kind == "jump_diffusion":
        return gen_jump_diffusion(rng, n), kind
    return gen_vol_cluster(rng, n), kind


# --- noise-robust: noisy context + clean conditional-mean horizon (Path B) ---

def gen_noise_robust(rng, ctx: int, horizon: int, *, kind: Optional[str] = None
                     ) -> Tuple[np.ndarray, str]:
    """Return a length ``ctx+horizon`` series: noisy realization on ``[0:ctx)`` and
    the clean conditional mean ``E[x_{t+h}|F_ctx]`` on ``[ctx:ctx+horizon)``.

    The forecastable part of each process is analytic, so the horizon is *exactly*
    the conditional mean — the model is trained to predict it while seeing the noisy
    context, and never to chase the unpredictable innovation."""
    kind = (_NOISE_ROBUST_KINDS[int(rng.integers(len(_NOISE_ROBUST_KINDS)))]
            if kind is None else kind)
    h = np.arange(1, horizon + 1, dtype=np.float64)

    if kind == "ou":
        theta = float(rng.uniform(0.05, 0.3)); mu = float(rng.uniform(-20, 20))
        sigma = float(rng.uniform(0.5, 4)); x0 = float(rng.uniform(-20, 20))
        ctx_x = gen_ou(rng, ctx, theta=theta, mu=mu, sigma=sigma, x0=x0)
        clean = mu + (ctx_x[-1] - mu) * np.exp(-theta * h)  # mean reverts to mu

    elif kind == "rw_drift":
        drift = float(rng.uniform(-0.3, 0.3)); sigma = float(rng.uniform(0.5, 3))
        ctx_x = np.cumsum(rng.normal(drift, sigma, ctx)) + rng.uniform(-20, 20)
        clean = ctx_x[-1] + drift * h                       # E[RW+drift] = last + drift·h

    elif kind == "gbm":
        mu = float(rng.uniform(-0.002, 0.003)); sigma = float(rng.uniform(0.005, 0.03))
        log0 = float(rng.uniform(2, 5))
        log_path = log0 + np.cumsum(rng.normal(mu, sigma, ctx))
        ctx_x = np.exp(log_path)
        clean = ctx_x[-1] * np.exp(mu * h)                  # E[S_{t+h}] = S_t·exp(μ·h)

    else:  # seasonal_ar — deterministic trend+season is forecastable; AR noise reverts
        usable = [p for p in _CALENDAR_PERIODS if p < ctx / 2] or [min(_CALENDAR_PERIODS)]
        period = int(rng.choice(usable)); amp = float(rng.uniform(2, 12))
        phase = float(rng.uniform(0, 2 * np.pi)); slope = float(rng.uniform(-0.1, 0.1))
        offset = float(rng.uniform(-20, 20))

        def signal(t):
            return offset + slope * t + amp * np.sin(2 * np.pi * t / period + phase)

        tc = np.arange(ctx, dtype=np.float64)
        ar_noise = S.gen_ar(rng, ctx, p=1, noise=float(rng.uniform(0.3, 1.5)))
        ctx_x = signal(tc) + ar_noise
        clean = signal(np.arange(ctx, ctx + horizon, dtype=np.float64))  # AR mean → 0

    return np.concatenate([ctx_x, clean]).astype(np.float64), kind


# --- structural TS -----------------------------------------------------------

def gen_structural_ts(rng, n: int) -> Tuple[np.ndarray, int]:
    """Trend × multi-seasonal (+ optional holiday spikes) + noise. Returns ``(x, m)``."""
    usable = [p for p in _CALENDAR_PERIODS if p < n / 2] or [min(_CALENDAR_PERIODS)]
    t = np.arange(n, dtype=np.float64)
    x = rng.uniform(-2, 2) * (t / max(1, n)) * rng.uniform(5, 50)  # trend
    k = int(rng.integers(1, 3)); seasons = rng.choice(usable, size=min(k, len(usable)), replace=False)
    amps = []
    for p in seasons:
        a = rng.uniform(2, 12); amps.append(a)
        x = x + a * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi))
    if rng.random() < 0.3:  # holiday/level spikes
        idx = rng.integers(0, n, size=int(rng.integers(1, 5)))
        x[idx] += rng.uniform(20, 80) * rng.choice([-1.0, 1.0])
    x = x + rng.normal(0, 1.0, n)
    dominant = int(seasons[int(np.argmax(amps))])
    return x, dominant


# --- dilution channels (D5/D13) ----------------------------------------------

def inject_dilution(rng, data: np.ndarray, nf: int, nt: int, *,
                    prob: float = 0.3, max_extra: int = 3,
                    weak_signal_frac: float = 0.5) -> Tuple[np.ndarray, int, int]:
    """With probability ``prob``, prepend 1..``max_extra`` extra *feature* channels
    (pure noise or weak-signal distractors) the model must learn to ignore. Targets
    are never touched; features-first ordering is preserved."""
    if prob <= 0 or rng.random() >= prob:
        return data, nf, nt
    C, n = data.shape
    n_extra = int(rng.integers(1, max_extra + 1))
    extras = []
    targets = data[nf:]
    for _ in range(n_extra):
        if rng.random() < weak_signal_frac and targets.shape[0] > 0:
            # weak-signal distractor: a target buried under heavy noise (faint, mostly useless)
            tgt = targets[int(rng.integers(targets.shape[0]))]
            base = np.nan_to_num(tgt, nan=0.0)
            w = float(rng.uniform(0.05, 0.2))
            scale = (np.nanstd(base) + 1e-6)
            extras.append(w * base + (1.0 - w) * scale * rng.normal(0, 1, n))
        else:
            # pure noise / unrelated channel salad
            if rng.random() < 0.5:
                extras.append(rng.normal(rng.uniform(-10, 10), rng.uniform(0.5, 5), n))
            else:
                extras.append(S.gen_primitive(rng, n))  # unrelated structured series
    extra_arr = np.stack(extras).astype(data.dtype)
    return np.concatenate([extra_arr, data], axis=0), nf + n_extra, nt


# --- family dispatch ---------------------------------------------------------

def gen_general_series(rng, n: int, family: str
                       ) -> Tuple[np.ndarray, int, int, int, str, int, int]:
    """One ``synth_general`` series. Returns
    ``(data[C,t], nf, nt, season_length, kind, crop_ctx, crop_p)`` where
    ``crop_ctx``/``crop_p`` are ``-1`` unless this is a fixed-window noise-robust item.
    """
    if family == "kernelsynth":
        C = int(rng.integers(1, 9))
        return gen_kernelsynth(rng, n, C), 0, C, -1, "kernelsynth", -1, -1
    if family == "sde":
        x, kind = gen_sde(rng, n)
        return x[None, :], 0, 1, -1, kind, -1, -1
    if family == "noise_robust":
        ctx = int(rng.integers(64, 513))
        horizon = int(rng.integers(8, 65))
        x, kind = gen_noise_robust(rng, ctx, horizon)
        return x[None, :], 0, 1, -1, f"nr_{kind}", ctx, horizon
    if family == "structural_ts":
        x, m = gen_structural_ts(rng, n)
        return x[None, :], 0, 1, m, "structural_ts", -1, -1
    if family == "multi_seasonal":
        from .synthetic_corpus import gen_multi_seasonal
        x, m = gen_multi_seasonal(rng, n)
        return x[None, :], 0, 1, m, "multi_seasonal", -1, -1
    if family == "shared_factor":
        C = int(rng.integers(2, 9))
        return S.gen_shared_factor(rng, n, C), 0, C, -1, "shared_factor", -1, -1
    if family == "regime_jump":
        return S.gen_regime_jump(rng, n)[None, :], 0, 1, -1, "regime_jump", -1, -1
    if family == "intermittent":
        return S.gen_intermittent(rng, n)[None, :], 0, 1, -1, "intermittent", -1, -1
    raise ValueError(f"unknown general family {family!r}")


def general_picker(weights: Optional[Dict[str, float]] = None):
    w = weights or GENERAL_WEIGHTS
    names = list(w.keys())
    p = np.array([w[k] for k in names], dtype=np.float64)
    if (p <= 0).all():
        raise ValueError("general family weights must have positive total")
    return names, p / p.sum()


def general_kinds() -> List[str]:
    return list(GENERAL_WEIGHTS.keys())


def write_general_corpus(
    writer,
    *,
    n_series: int,
    seed: int = 0,
    length_range: Tuple[int, int] = (96, 4096),
    nan_prob: float = 0.25,
    nan_cap: float = 0.2,
    dilution_prob: float = 0.3,
    dilution_max_extra: int = 3,
    weights: Optional[Dict[str, float]] = None,
    source: str = "synth_general",
) -> int:
    """Generate ``n_series`` ``synth_general`` series and feed them to ``writer``.

    Deterministic: series ``idx`` keyed by ``(seed, marker, idx)``. Noise-robust
    items carry their ``(ctx, horizon)`` crop hint into the shard index and are
    **not** NaN-corrupted (their clean horizon must stay clean); other families get
    NaNs w.p. ``nan_prob``. Dilution feature channels are injected w.p.
    ``dilution_prob``. Returns the number of series written."""
    names, p = general_picker(weights)
    lo, hi = int(length_range[0]), int(length_range[1])
    for idx in range(int(n_series)):
        rng = np.random.default_rng((int(seed), 0x6234, idx))
        family = names[int(rng.choice(len(names), p=p))]
        n = int(rng.integers(lo, hi + 1))
        data, nf, nt, m, kind, crop_ctx, crop_p = gen_general_series(rng, n, family)
        is_nr = crop_ctx > 0
        data, nf, nt = inject_dilution(rng, np.atleast_2d(data), nf, nt,
                                       prob=dilution_prob, max_extra=dilution_max_extra)
        if not is_nr and nan_cap > 0 and rng.random() < nan_prob:
            data = S.inject_nans(rng, data, nan_cap)
        data = np.ascontiguousarray(np.atleast_2d(data), dtype=np.float32)
        writer.add(data, nf, nt, season_length=m, source=source, kind=kind,
                   item_id=f"general_{idx}", crop_ctx=crop_ctx, crop_p=crop_p)
    return int(n_series)
