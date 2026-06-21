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

import math
import time
from typing import List, Optional, Tuple

import torch

from ..config import Config
from ..losses import LossBreakdown
from ..model.tetris import Tetris
from ..packing.reservoir import StreamingReservoir, packed_batches
from .step import make_basis, mark_dynamic_batch, train_step
from .tracking import (
    NullTracker, Tracker, eval_scalars, eval_tail_scalars, per_config_rows,
)


def _build_scheduler(optimizer, cfg, *, total_steps: int):
    """LR schedule (Tier-1): linear warmup over ``run.warmup_steps`` then
    ``run.lr_schedule`` ∈ {constant, cosine} (cosine decays peak→0 across the
    remaining steps). Defaults (warmup=0, constant) ⇒ a no-op constant LR, so the
    old behavior is exactly preserved. Returns a ``LambdaLR`` stepped per train step."""
    warmup = max(0, int(getattr(cfg.run, "warmup_steps", 0)))
    schedule = getattr(cfg.run, "lr_schedule", "constant")
    total = max(1, int(total_steps))

    def lr_factor(s: int) -> float:      # s = scheduler.step() count (0 on first call)
        if warmup > 0 and s < warmup:
            return (s + 1) / warmup
        if schedule == "cosine":
            prog = min(1.0, max(0.0, (s - warmup) / max(1, total - warmup)))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def _step_scalars(lb: LossBreakdown, ema: float, lr: float, rate: float) -> dict:
    """Flatten one step's loss breakdown + telemetry into scalars (G1).

    Core training scalars (train_loss/ema/lr/throughput + loss components) are kept
    **unconditionally** — a divergence must surface as an ``inf`` spike, not a
    silently dropped point. The optional grad-norm + horizon ``diag`` extras are
    finite-filtered so a stray NaN diagnostic never poisons a tracker panel."""
    core = {
        "train_loss": float(lb.total.detach()),
        "train_loss_ema": float(ema),
        "lr": lr,
        "steps_per_sec": rate,
        "loss/horizon": float(lb.horizon.detach()),
        "loss/aux_total": float(lb.aux_total.detach()),
    }
    for k, a in enumerate(lb.aux_per_tier):
        core[f"loss/aux_tier_{k}"] = float(a.detach())
    extras: dict = {}
    if lb.grad_norm is not None:
        extras["grad_norm"] = float(lb.grad_norm)
    if lb.diag:
        extras.update(lb.diag)
    core.update((k, v) for k, v in extras.items()
                if isinstance(v, (int, float)) and math.isfinite(v))
    return core


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
    log_every: int = 0,
    on_log=None,
    on_eval=None,
    tracker: Optional[Tracker] = None,
) -> List[float]:
    """Run up to ``steps`` train steps off the reservoir; return per-step loss.

    ``forward``/``model`` default to a fresh eager :class:`Tetris` (pass a compiled
    forward + its model to exercise the compiled path). Stops early if the loader
    drains before ``steps``.

    Optional **record-only** eval (§6, S12): when ``eval_every > 0`` and an
    ``eval_loader`` is given, the GIFT-Eval test loss is computed every
    ``eval_every`` steps via the *shared* collator (context-only) and appended to
    ``eval_log`` as ``(step, loss)`` — it never feeds the optimizer.

    ``tracker`` (G1): an optional experiment tracker (default :class:`NullTracker`);
    train_loss/lr/throughput are logged each ``log_every`` and the eval result each
    ``eval_every`` — a no-op sink unless the caller wires a backend.
    """
    if tracker is None:
        tracker = NullTracker()
    if model is None:
        model = Tetris(cfg).to(device)
    if forward is None:
        forward = model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=getattr(cfg.run, "weight_decay", 0.0))
    scheduler = _build_scheduler(optimizer, cfg, total_steps=steps)
    if reservoir is None:
        reservoir = StreamingReservoir.from_cfg(cfg, rank=rank, world_size=world_size)

    batches = packed_batches(
        reservoir,
        l_pack=cfg.packing.L_pack,
        p_out=cfg.model.out_patch,
        num_buffers=cfg.packing.buffers_per_step,
    )

    losses: List[float] = []
    # Geometric (log-space) EMA — early arcsinh-space losses span many orders of
    # magnitude (random-init real-space error can be ~1e8); a linear EMA gets
    # permanently poisoned by that transient, a log-space one rides through it.
    ema_log: Optional[float] = None
    t0 = time.perf_counter()
    for batch in batches:
        if len(losses) >= steps:
            break
        batch = batch.to(device)
        basis = make_basis(batch, cfg.model.d_model, device=device, generator=generator)
        mark_dynamic_batch(batch, basis)
        step = len(losses) + 1
        want_diag = log_every > 0 and step % log_every == 0   # only pay diag cost on log steps
        lb = train_step(
            forward, batch, basis, optimizer,
            aux_weights=cfg.loss.aux_weights, loss_space=cfg.norm.loss_space,
            max_grad_norm=cfg.run.grad_clip, collect_diag=want_diag,
        )
        scheduler.step()                 # advance LR (warmup → cosine/constant)
        losses.append(float(lb.total.detach()))
        if math.isfinite(losses[-1]):    # never let an inf/nan poison the EMA
            ll = math.log1p(max(losses[-1], 0.0))
            ema_log = ll if ema_log is None else 0.98 * ema_log + 0.02 * ll

        # Live train-loss heartbeat (fires between evals so long runs log progress).
        if log_every > 0 and step % log_every == 0:
            if on_log is not None:
                on_log(step, losses[-1])
            rate = step / max(1e-9, time.perf_counter() - t0)
            ema = math.expm1(ema_log) if ema_log is not None else losses[-1]
            cur_lr = optimizer.param_groups[0]["lr"]   # log the live scheduled LR
            tracker.log_scalars(_step_scalars(lb, ema, cur_lr, rate), step=step)
        if eval_loader is not None and eval_every > 0 and step % eval_every == 0:
            fn = eval_fn
            if fn is None:
                from ..data.eval_loader import evaluate_test_loss

                fn = evaluate_test_loss
            tl = fn(forward, eval_loader, cfg, device=device)
            if eval_log is not None:
                eval_log.append((step, tl))
            if on_eval is not None:           # live eval logging (not deferred to end)
                on_eval(step, tl, losses[-1])
            tracker.log_scalars(eval_scalars(tl), step=step)
            tail = eval_tail_scalars(tl)      # tail/blowup summary (leaderboard evals)
            if tail:
                tracker.log_scalars(tail, step=step)
                cols, rows = per_config_rows(tl)
                tracker.log_table("eval/per_config", cols, rows, step=step)
    return losses
