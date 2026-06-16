"""One train step (S10) — forward + loss + backward + optimizer.

The compiled region is the model forward; loss/backward/opt stay eager. Two
quantities vary per step and would otherwise force recompiles (D14): the raw-store
width ``R`` (``norm_values``/``observed``) and the variate count ``n_var`` (the
basis). :func:`mark_dynamic_batch` marks those dims dynamic so **one graph serves
all data**. The variate basis (D4) is sampled **eagerly here** (QR) and passed in,
keeping the orthonormalization out of the graph.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..backend import backend_kind
from ..losses import LossBreakdown, compute_loss
from ..masks import build_block_mask, build_sdpa_mask
from ..model.variate_id import sample_orthonormal_basis


def make_basis(
    batch,
    d_model: int,
    *,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Eager per-buffer orthonormal ID pool sized to this batch's variate count."""
    n_var = max(int(batch.variate_uid.max().item()) + 1, 1)
    return sample_orthonormal_basis(batch.B, n_var, d_model, device=device, generator=generator)


def mark_dynamic_batch(batch, basis: torch.Tensor) -> None:
    """Mark the only per-step-varying dims dynamic (D14 one-graph): ``R`` and ``n_var``."""
    torch._dynamo.mark_dynamic(batch.norm_values, 1)
    torch._dynamo.mark_dynamic(batch.observed, 1)
    torch._dynamo.mark_dynamic(basis, 1)


def make_block_mask(batch):
    """Build the attention mask **eagerly** for the hoist out of the compiled
    forward (D14): a FlexAttention ``BlockMask`` can't be constructed inside
    ``torch.compile`` (inductor can't lower ``create_block_mask``). Flex on CUDA,
    materialized bool mask on CPU/MPS."""
    if backend_kind(batch.sample_id.device) == "flex":
        return build_block_mask(batch.sample_id, batch.role, batch.t_center)
    return build_sdpa_mask(batch.sample_id, batch.role, batch.t_center)


def train_step(
    forward,
    batch,
    basis: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    aux_weights: Sequence[float],
    loss_space: str = "arcsinh",
    max_grad_norm: float = 0.0,
) -> LossBreakdown:
    """Run one optimization step; returns the loss breakdown (D6). The block mask is
    built eagerly here and passed in (hoisted out of the compiled forward)."""
    out = forward(batch, variate_basis=basis, block_mask=make_block_mask(batch))
    lb = compute_loss(out, batch, aux_weights=aux_weights, loss_space=loss_space)
    optimizer.zero_grad(set_to_none=True)
    lb.total.backward()
    if max_grad_norm and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for grp in optimizer.param_groups for p in grp["params"]], max_grad_norm)
    optimizer.step()
    return lb
