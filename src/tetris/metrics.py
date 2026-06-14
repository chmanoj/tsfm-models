"""Metrics (D13) — v1 tracks **test loss only**, record-only.

The decision signal in v1 is the train-split validation windows; the GIFT-Eval
test-split loss on the deterministic first-100-windows-per-config shard is recorded
but not used for decisions. **MASE is deferred (O4)**: seasonal-period detection +
seasonal-naive denominators + the true MASE formula land in a later iteration; the
stubs below exist so wiring them is purely additive.
"""

from __future__ import annotations

from typing import Optional

import torch

from .losses import horizon_loss


@torch.no_grad()
def horizon_test_loss(output, batch) -> float:
    """Record-only horizon MAE on a held-out shard (Stage 9 space, detached)."""
    return float(horizon_loss(output.horizon, batch))


def seasonal_naive_denom(*args, **kwargs) -> Optional[torch.Tensor]:
    """O4 deferred — seasonal-naive denominator for MASE. ``None`` in v1."""
    return None


def mase(*args, **kwargs):  # pragma: no cover - O4 deferred
    """O4 deferred — true-formula MASE (needs seasonal-period detection)."""
    raise NotImplementedError("MASE deferred to a later iteration (O4)")
