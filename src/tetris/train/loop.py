"""Training loop over the streaming reservoir (S11).

The reservoir path: ``build_loader`` → :class:`StreamingReservoir` (window-sample +
reservoir + best-fit-decreasing + cost-bucketing) → :func:`packed_batches` (the
frozen ``pack``) → ``train_step``. Packing happens in the ``packed_batches``
adapter, **outside** this loop — the loop only moves the batch to the device,
hoists the eager variate basis, marks the dynamic dims, and steps. It reuses the
*same* ``pack``/``train_step`` as the S10 trivial path; only the grouping changes
(``cfg.packing.reservoir`` flips on). Stays rank-local (O6); DDP/FSDP wrap +
checkpoint re-shard are S13.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from ..config import Config
from ..model.tetris import Tetris
from ..packing.reservoir import StreamingReservoir, packed_batches
from .step import make_basis, mark_dynamic_batch, train_step


def run_training(
    cfg: Config,
    *,
    steps: int,
    lr: float = 1e-3,
    device: str = "cpu",
    rank: int = 0,
    world_size: int = 1,
    forward=None,
    model: Optional[Tetris] = None,
    generator: Optional[torch.Generator] = None,
    reservoir: Optional[StreamingReservoir] = None,
    eval_loader=None,
    eval_every: int = 0,
    eval_log: Optional[List[Tuple[int, float]]] = None,
    eval_fn=None,
) -> List[float]:
    """Run up to ``steps`` train steps off the reservoir; return per-step loss.

    ``forward``/``model`` default to a fresh eager :class:`Tetris` (pass a compiled
    forward + its model to exercise the compiled path). Stops early if the loader
    drains before ``steps``.

    Optional **record-only** eval (§6, S12): when ``eval_every > 0`` and an
    ``eval_loader`` is given, the GIFT-Eval test loss is computed every
    ``eval_every`` steps via the *shared* collator (context-only) and appended to
    ``eval_log`` as ``(step, loss)`` — it never feeds the optimizer.
    """
    if model is None:
        model = Tetris(cfg).to(device)
    if forward is None:
        forward = model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if reservoir is None:
        reservoir = StreamingReservoir.from_cfg(cfg, rank=rank, world_size=world_size)

    batches = packed_batches(
        reservoir,
        l_pack=cfg.packing.L_pack,
        p_out=cfg.model.out_patch,
        num_buffers=cfg.packing.buffers_per_step,
    )

    losses: List[float] = []
    for batch in batches:
        if len(losses) >= steps:
            break
        batch = batch.to(device)
        basis = make_basis(batch, cfg.model.d_model, device=device, generator=generator)
        mark_dynamic_batch(batch, basis)
        lb = train_step(
            forward, batch, basis, optimizer,
            aux_weights=cfg.loss.aux_weights, loss_space=cfg.norm.loss_space,
        )
        losses.append(float(lb.total.detach()))

        step = len(losses)
        if eval_loader is not None and eval_every > 0 and step % eval_every == 0:
            fn = eval_fn
            if fn is None:
                from ..data.eval_loader import evaluate_test_loss

                fn = evaluate_test_loss
            tl = fn(forward, eval_loader, cfg, device=device)
            if eval_log is not None:
                eval_log.append((step, tl))
    return losses
