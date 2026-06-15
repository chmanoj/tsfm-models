"""D10 — per-channel anchored-arcsinh normalization (reversible) + locally
re-anchored loss targets.

Input representation (per channel, per sample, context statistics only):

    z = arcsinh((x - a) / sigma_delta)

with anchor ``a = median(last N raw steps)`` (N tunable, 8–16 for steep trends)
and robust step-scale ``sigma_delta = 1.4826 * median|dx|`` with the fallback
chain ``-> mean|dx| -> level-IQR/1.349 -> 1`` plus a relative floor
``sigma_delta >= 1e-3 * IQR/1.349``. Exact inversion ``x = sinh(z)*sigma + a``
denormalizes horizon predictions (no leakage — context-only statistics).

Coarse tiers carry their own per-tier ``sigma_delta`` (robust median |delta| of
the tier-aggregated sequence; D10 + D12 six tiers).

Loss targets are *locally re-anchored* (D10/D6): every prediction target — aux
and horizon — is movement from the prediction point's local level in step-vol
units, arcsinh-compressed, so loss sensitivity is uniform across positions and
regimes. The ``loss_target=global_norm_space`` hedge swaps these for global-z
targets via a one-line config change (handled in ``losses.py``).

All functions operate per channel on a 1-D tensor (the collator loops channels).
Statistics are computed in float64 for stability and cast back to the input
dtype; ``forward``/``invert`` are dtype-preserving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .constants import PATCH, V

# Gaussian consistency factors so all robust scales estimate the same sigma:
# MAD = 0.6745 sigma -> sigma = 1.4826 * MAD ; IQR = 1.349 sigma.
_MAD_TO_SIGMA = 1.4826
_IQR_TO_SIGMA = 1.0 / 1.349
_REL_FLOOR = 1e-3  # sigma_delta >= 1e-3 * IQR/1.349


@dataclass
class ChannelStats:
    """Per-(sample,channel) normalization statistics (D10 / walkthrough Stage 3).
    One anchor ``a`` and a per-tier scale vector ``sigma_delta[V]``, constant across
    all timesteps (not rolling/per-step — this is what makes 2-number inversion work).

    - ``sigma_delta[0]`` is the raw base ``1.4826·median|Δx|``; it is the scale used
      for the input ``norm_values`` and for Stage-9 horizon inversion.
    - ``sigma_delta[1..5]`` are per-tier scales (robust σ of each tier's aggregated
      sequence), used for the locally-reanchored aux targets (D10) and receipts (D4).

    ``fallback`` records which rung of the base sigma chain fired (diagnostic)."""

    a: torch.Tensor            # scalar
    sigma_delta: torch.Tensor  # [V]
    fallback: str

    @property
    def sigma(self) -> torch.Tensor:
        """The base scale used for input normalization / inversion = sigma_delta[0]."""
        return self.sigma_delta[0]


def _finite(v: torch.Tensor) -> torch.Tensor:
    """Drop NaN/inf entries from a 1-D tensor."""
    return v[torch.isfinite(v)]


def _nanmedian(v: torch.Tensor) -> torch.Tensor:
    v = _finite(v)
    if v.numel() == 0:
        return torch.tensor(float("nan"), dtype=v.dtype)
    return torch.median(v)


def _iqr(levels: torch.Tensor) -> torch.Tensor:
    """Interquartile range of observed levels (NaN-dropped). 0 if < 2 points."""
    v = _finite(levels)
    if v.numel() < 2:
        return torch.zeros((), dtype=levels.dtype)
    q = torch.quantile(v, torch.tensor([0.25, 0.75], dtype=v.dtype))
    return q[1] - q[0]


def _robust_sigma(levels: torch.Tensor, diffs: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Robust step-scale with D10 fallback chain + relative floor.

    Returns (sigma, fallback_label). ``fallback_label`` names the rung used.
    """
    adx = _finite(diffs).abs()

    fallback = "mad"
    if adx.numel() > 0:
        sigma = _MAD_TO_SIGMA * torch.median(adx)
    else:
        sigma = torch.tensor(float("nan"), dtype=levels.dtype)

    if not (sigma > 0):
        # intermittent / quantized: median|dx| degenerate -> mean|dx|
        fallback = "mean_abs"
        sigma = adx.mean() if adx.numel() > 0 else torch.tensor(float("nan"), dtype=levels.dtype)

    if not (sigma > 0):
        # spread of levels (robust)
        fallback = "iqr"
        sigma = _iqr(levels) * _IQR_TO_SIGMA

    if not (sigma > 0):
        # genuinely constant series -> degrade gracefully to all-zeros
        fallback = "unit"
        sigma = torch.ones((), dtype=levels.dtype)

    # Relative floor (only raises sigma; never lowers it).
    iqr = _iqr(levels)
    if iqr > 0:
        floor = _REL_FLOOR * iqr * _IQR_TO_SIGMA
        sigma = torch.maximum(sigma, floor.to(sigma.dtype))

    return sigma, fallback


def _tier_aggregate(x: torch.Tensor, patch: int) -> torch.Tensor:
    """Mean-aggregate a 1-D series into non-overlapping patches of ``patch``
    steps (NaN-aware; trailing partial patch dropped). Anchored at the most
    recent step so tier boundaries align with the forecast origin (D2)."""
    n = x.numel()
    full = (n // patch) * patch
    if full == 0:
        return x.new_empty(0)
    # Keep the most-recent ``full`` steps (drop the oldest partial patch).
    xw = x[n - full:].reshape(-1, patch)
    mask = torch.isfinite(xw)
    counts = mask.sum(dim=1)
    summed = torch.where(mask, xw, torch.zeros_like(xw)).sum(dim=1)
    agg = summed / counts.clamp(min=1)
    agg[counts == 0] = float("nan")  # fully-missing patch -> NaN (D7 [NA] upstream)
    return agg


def compute_stats(x: torch.Tensor, anchor_window: int = 32) -> ChannelStats:
    """Compute per-channel anchored-arcsinh statistics from context values.

    ``x``: 1-D context series (raw, may contain NaN). Uses context statistics
    only (no leakage). Computed in float64, returned in ``x``'s dtype.
    """
    in_dtype = x.dtype
    xf = x.to(torch.float64)

    # Anchor a = median of last N observed raw steps.
    n = xf.numel()
    w = min(anchor_window, n) if n > 0 else 0
    tail = xf[n - w:] if w > 0 else xf
    a = _nanmedian(tail)
    if not torch.isfinite(a):
        a = _nanmedian(xf)
    if not torch.isfinite(a):
        a = torch.zeros((), dtype=torch.float64)

    diffs = xf[1:] - xf[:-1] if n >= 2 else xf.new_empty(0)
    sigma_base, fallback = _robust_sigma(xf, diffs)

    # sigma_delta[0] = raw base (input scale); [1..5] = per-tier aggregated scales,
    # falling back to the base when a tier has too few aggregated points.
    sigma_delta: List[torch.Tensor] = [sigma_base]
    for p in PATCH[1:]:
        agg = _tier_aggregate(xf, p)
        if agg.numel() >= 2:
            d = agg[1:] - agg[:-1]
            s_k, _ = _robust_sigma(agg, d)
            if not (s_k > 0):
                s_k = sigma_base
        else:
            s_k = sigma_base
        sigma_delta.append(s_k)

    return ChannelStats(
        a=a.to(in_dtype),
        sigma_delta=torch.stack(sigma_delta).to(in_dtype),
        fallback=fallback,
    )


def forward(x: torch.Tensor, a: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Anchored-arcsinh transform: ``z = arcsinh((x - a) / sigma)``.
    Dtype-preserving; NaN propagates (missingness handled upstream by D7)."""
    return torch.arcsinh((x - a) / sigma)


def invert(z: torch.Tensor, a: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`forward`: ``x = sinh(z) * sigma + a``."""
    return torch.sinh(z) * sigma + a


# Safety clamp for the arcsinh→raw inversion at the *forecast/metric* boundary
# (G3.1). `sinh` grows exponentially, so an unbounded arcsinh-space prediction
# overflows to ±inf (sinh(89) > float32 max), poisoning MASE and, in rollout,
# cascading NaN. Clamping the prediction to ±10 keeps the exponential bounded in
# *every* common precision — sinh(10) ≈ 11013 < fp16 max (65504) < bf16/fp32 max —
# while never clipping a legitimate prediction: real arcsinh-space outputs live
# around 0–7 (arcsinh of hundreds of σ), and ±10 corresponds to ~11000·σ. This is
# applied only at the denorm boundary; the math primitives above stay exact.
ARCSINH_INV_CLAMP = 10.0


# --- Locally re-anchored loss targets (D10/D6) ------------------------------

def horizon_target(y: torch.Tensor, a: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Horizon loss target: ``arcsinh((y - a)/sigma)`` — the anchor ``a`` *is* the
    forecast origin's local level, so this is the same form as the input map."""
    return torch.arcsinh((y - a) / sigma)


def horizon_invert(z: torch.Tensor, a: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Denormalize a horizon prediction back to raw value space."""
    return torch.sinh(z) * sigma + a


def aux_target(future: torch.Tensor, level: torch.Tensor, sigma_tier: torch.Tensor) -> torch.Tensor:
    """Aux (next-patch) loss target for a tier token: movement from the
    prediction point's local ``level`` in tier step-vol units, arcsinh-compressed:
    ``arcsinh((future - level)/sigma_tier)`` (D10 local re-anchoring)."""
    return torch.arcsinh((future - level) / sigma_tier)


def global_norm_target(values: torch.Tensor, a: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The ``loss_target=global_norm_space`` hedge (D10): targets live in the
    same global anchored-z space as the input. One-line fallback if locally
    re-anchored targets prove hard to learn."""
    return forward(values, a, sigma)


__all__ = [
    "ChannelStats",
    "compute_stats",
    "forward",
    "invert",
    "horizon_target",
    "horizon_invert",
    "aux_target",
    "global_norm_target",
    "PATCH",
    "V",
]
