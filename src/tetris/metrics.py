"""Metrics (D13) — record-only horizon MAE + MASE vs seasonal naive.

The record-only test loss (``horizon_test_loss``) is the Stage-9 normalized horizon
MAE on a held-out shard. **MASE (O4) is now wired** for the sanity / GIFT-Eval
scoring: the GIFT-Eval / gluonts formula ``MAE_horizon / in-sample seasonal-naive
denom``, where the season length ``m`` is **dataset metadata** (carried on the eval
item, never detected by the model). All three helpers work in **raw value space**
(1-D per channel) so the model's forecast must be inverted out of normalized space
(``sinh(·)·σ+a``) before scoring — exactly the space the leaderboard reports.
"""

from __future__ import annotations

import torch

from .losses import horizon_loss

_DENOM_FLOOR = 1e-8


@torch.no_grad()
def horizon_test_loss(output, batch) -> float:
    """Record-only horizon MAE on a held-out shard (Stage 9 space, detached)."""
    return float(horizon_loss(output.horizon, batch))


def _impute_last_value(x: torch.Tensor) -> torch.Tensor:
    """gluonts ``LastValueImputation``: replace each NaN with the last finite value
    (forward-fill); leading NaN -> the first finite value; an all-NaN (or length-1
    all-NaN) series -> 0 (``DummyValueImputation`` default). Pure-torch replica so
    ``metrics`` keeps no gluonts dependency."""
    x = x.reshape(-1).to(torch.float32).clone()
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return torch.zeros_like(x)
    n = x.numel()
    pos = torch.arange(n)
    idx = torch.cummax(torch.where(finite, pos, torch.zeros_like(pos)), dim=0).values
    out = x[idx]                                   # last finite value at/<= each index
    first = int(torch.nonzero(finite, as_tuple=True)[0][0])
    if first > 0:
        out[:first] = x[first]                     # leading NaN -> first finite value
    return out


def seasonal_naive_forecast(context: torch.Tensor, m: int, p: int) -> torch.Tensor:
    """Seasonal-naive horizon forecast, matching GIFT-Eval's gluonts
    ``SeasonalNaivePredictor`` (so the baseline is scored exactly as the leaderboard
    does): impute missing values in the in-sample target ``context`` via last-value
    imputation, then repeat the trailing season of length ``m`` over ``p`` steps. If
    the context is shorter than one season, fall back to the mean of the finite
    observations (gluonts ``np.nanmean``). The imputation is what keeps the baseline
    finite on the ``*_with_missing`` / ``bitbrains`` configs (NaN at the seasonal
    lag would otherwise NaN the forecast and poison the snaive/skill aggregate)."""
    context = context.reshape(-1).to(torch.float32)
    n = context.numel()
    if n == 0:
        return torch.zeros(p, dtype=torch.float32)
    season = max(1, int(m))
    if n >= season:
        target = _impute_last_value(context)
        idx = (n - season) + (torch.arange(p) % season)
        return target[idx].to(torch.float32)
    # series shorter than one season: gluonts uses the (NaN-aware) mean
    finite = context[torch.isfinite(context)]
    fill = float(finite.mean()) if finite.numel() else 0.0
    return torch.full((p,), fill, dtype=torch.float32)


def seasonal_naive_denom(context: torch.Tensor, m: int) -> torch.Tensor:
    """In-sample seasonal-naive denominator for MASE: ``mean_t |y_t - y_{t-m}|``
    over the context (1-D, raw), floored to avoid divide-by-zero.

    **Data** NaNs in the context are masked out (mean over the finite seasonal
    diffs), mirroring gluonts' ``Evaluator`` (``np.ma.masked_invalid(past_data)``)
    so missing observations in the ``*_with_missing`` configs don't NaN the denom.
    Model output is never involved here, so this never hides a model failure."""
    context = context.reshape(-1).to(torch.float32)
    m = max(1, int(m))
    if context.numel() <= m:
        return torch.tensor(_DENOM_FLOOR)
    diff = (context[m:] - context[:-m]).abs()
    finite = diff[torch.isfinite(diff)]
    if finite.numel() == 0:
        return torch.tensor(_DENOM_FLOOR)
    return finite.mean().clamp_min(_DENOM_FLOOR)


def mase(y_true: torch.Tensor, y_pred: torch.Tensor, denom: torch.Tensor) -> float:
    """Mean Absolute Scaled Error: horizon MAE divided by the in-sample
    seasonal-naive ``denom`` (raw space, 1-D)."""
    mae = (y_true.reshape(-1).to(torch.float32) - y_pred.reshape(-1).to(torch.float32)).abs().mean()
    return float(mae / denom.clamp_min(_DENOM_FLOOR))
