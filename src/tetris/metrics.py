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


def seasonal_naive_forecast(context: torch.Tensor, m: int, p: int) -> torch.Tensor:
    """Seasonal-naive horizon forecast (gluonts): repeat the last season ``m`` of
    the in-sample target ``context`` (1-D, raw) over ``p`` steps. Falls back to the
    last value when the context is shorter than one season."""
    context = context.reshape(-1)
    if context.numel() == 0:
        return torch.zeros(p, dtype=torch.float32)
    m = max(1, min(int(m), context.numel()))
    last_season = context[-m:]
    idx = torch.arange(p) % m
    return last_season[idx].to(torch.float32)


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
