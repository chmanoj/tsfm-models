"""Losses (D6/D10, Stage 9 reduction).

Dense, fixed-shape reductions — heads run over all ``L`` and selection happens
here (no token gather before the head), so one compile graph.

- **Horizon:** MAE (median-optimal, D6) over ``[B,L,P_out]`` masked by
  ``target_valid`` (NaN GT and non-query slots dropped). Space = anchored-arcsinh
  base scale (``horizon_target`` is already in that space).
- **Aux next-patch:** six dense tier-heads ``[B,L,P_k]``; each token tier-selects
  its own head, compared to the next ``P_k`` raw steps ``[t+P_k, t+2P_k)`` gathered
  from ``norm_values`` via ``raw_start``, masked by ``valid_aux`` **and** the target
  steps' observed bit. Per-tier ``aux_weights[6]`` replace a single λ.
  ``total = horizon_MAE + Σ_k aux_weights[k]·aux_MAE_k``.

**Deferred (v1):** aux targets use the **base scale** ``σΔ[0]`` (gathered from the
Batch's ``norm_values``), *not* D10's per-tier locally-reanchored ``σΔ[1..5]`` —
honoring that would reopen the frozen S6 Batch (decided: keep base-scale; see the
post-S8 reconciliation note). ``loss_target`` (config) is therefore not yet wired;
``loss_space`` honors ``arcsinh`` (default), ``vol_units`` is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from .constants import PATCH, V


@dataclass
class LossBreakdown:
    total: torch.Tensor               # scalar
    horizon: torch.Tensor             # scalar
    aux_total: torch.Tensor           # scalar (Σ_k w_k · aux_k)
    aux_per_tier: List[torch.Tensor]  # 6 × scalar (unweighted aux MAE per tier)
    # Training-step telemetry (G1 per-step logging), filled by ``train_step`` — not
    # part of the loss math. ``grad_norm`` is the pre-clip global grad norm; ``diag``
    # is an optional dict of horizon diagnostics (z-space + real-space, composition).
    grad_norm: Optional[torch.Tensor] = None
    diag: Optional[dict] = None


def _masked_mae(err_abs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(|·|·mask).sum() / mask.sum()`` with a denominator floor (D6/D7)."""
    denom = mask.sum().clamp_min(1).to(err_abs.dtype)
    return (err_abs * mask).sum() / denom


def horizon_loss(horizon_pred: torch.Tensor, batch) -> torch.Tensor:
    """Dense horizon MAE masked by ``target_valid`` (Stage 9, option-β)."""
    valid = batch.target_valid.to(horizon_pred.dtype)
    return _masked_mae((horizon_pred - batch.horizon_target).abs(), valid)


def aux_loss_per_tier(aux_pred: Sequence[torch.Tensor], batch) -> List[torch.Tensor]:
    """Unweighted next-patch MAE for each of the six tiers (base scale, v1).

    For tier ``k``, gathers the target window ``[raw_start+P_k, raw_start+2P_k)``
    from ``norm_values`` and masks by ``tier_id==k & valid_aux`` and the target
    steps' ``observed`` bit (and in-range guard)."""
    B, L = batch.tier_id.shape
    R = batch.norm_values.shape[1]
    out: List[torch.Tensor] = []
    for k in range(V):
        p_k = PATCH[k]
        base = batch.raw_start.long() + p_k                       # [B, L]
        offs = torch.arange(p_k, device=base.device)
        gidx = base.unsqueeze(-1) + offs                          # [B, L, P_k]
        in_range = (gidx >= 0) & (gidx < R)
        gflat = gidx.clamp(0, R - 1).reshape(B, L * p_k)
        tgt = torch.gather(batch.norm_values, 1, gflat).reshape(B, L, p_k)
        obs = torch.gather(batch.observed.to(torch.bool), 1, gflat).reshape(B, L, p_k)

        token_mask = (batch.tier_id == k) & batch.valid_aux       # [B, L]
        elem_mask = (token_mask.unsqueeze(-1) & obs & in_range).to(aux_pred[k].dtype)
        out.append(_masked_mae((aux_pred[k] - tgt).abs(), elem_mask))
    return out


def compute_loss(
    output,
    batch,
    *,
    aux_weights: Sequence[float],
    loss_space: str = "arcsinh",
) -> LossBreakdown:
    """Full training loss = horizon MAE + Σ_k ``aux_weights[k]``·aux_MAE_k."""
    if loss_space != "arcsinh":
        raise NotImplementedError(
            f"loss_space={loss_space!r} deferred; only 'arcsinh' is wired in v1"
        )
    assert len(aux_weights) == V, (len(aux_weights), V)

    h = horizon_loss(output.horizon, batch)
    per_tier = aux_loss_per_tier(output.aux, batch)
    w = torch.as_tensor(list(aux_weights), dtype=h.dtype, device=h.device)
    aux_total = sum((w[k] * per_tier[k] for k in range(V)), start=torch.zeros((), dtype=h.dtype))
    return LossBreakdown(total=h + aux_total, horizon=h, aux_total=aux_total, aux_per_tier=per_tier)
