"""GIFT-Eval ``test``-split overfit loader (G3/G3.1 — in-distribution capacity probe).

Streams each GIFT-Eval ``test`` window as **one continuous series** —
``cat(context, held-out horizon)`` of shape ``[n, t_ctx + p]`` — as a frozen
:class:`~tetris.data.contract.Item`, so the training window-sampler crops its own
ctx/horizon windows from the *whole* series (exactly like ``SanityTrainLoader``).
This is a deliberate **memorization-capacity overfit** (G3.1): to overfit the real
data the model must *train on the horizon it is later scored on*, so we merge the
held-out ``y_true`` (target rows) and ``feature_future`` (KFF feature rows, else NaN
for unknown future covariates) back into the series here. It is therefore **NOT
zero-shot** (the decision log's target) and not the no-leakage posture of the eval
path — the eval loader (``GiftEvalEvalLoader``) keeps the context/horizon split and
scores the held-out steps; only this *training* loader merges them.

Series are materialized **once** (the test set is finite) and then **cycled**
(overfit; the run stops at ``steps``), sharded disjointly by ``(rank, world_size)``
round-robin over the flattened item stream (O6 — no-op at ``world_size=1``).

The per-config window cap is ``cfg.eval.items_per_config`` (the universal per-config
cap from G2; ``-1`` -> all) and the GIFT-Eval terms come from ``cfg.data.terms``
(``short``/``medium``/``long``; all attempted, non-applicable pairs skipped):
training overfits on exactly the first-N test windows per ``(config, term)`` that the
leaderboard then scores, so the probe is honest.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

import torch

from .contract import Item, EvalItem, validate_item

log = logging.getLogger("tetris.gifteval")


def merge_context_horizon(e: EvalItem) -> Item:
    """Build one continuous training series ``cat(context, horizon)`` from an
    :class:`EvalItem` (G3.1 overfit). Shape ``[C, t_ctx + p]``: target rows get the
    context targets followed by the held-out ``y_true``; feature rows get the context
    features followed by ``feature_future`` when revealed (KFF) **else NaN** (the
    future covariate is unknown; the tokenizer masks NaN). The training sampler then
    crops ctx+horizon windows spanning the real target region — the point of an
    overfit. Returns the frozen ``(data_tensor, num_features, num_targets)`` Item."""
    ctx = e.data_tensor                                   # [C, t_ctx]
    nf, nt = e.num_features, e.num_targets
    C, t_ctx = ctx.shape
    p = int(e.y_true.shape[0])
    full = torch.full((C, t_ctx + p), float("nan"), dtype=torch.float32)
    full[:, :t_ctx] = ctx
    full[nf:, t_ctx:] = e.y_true.to(torch.float32).T      # [p, nt] -> [nt, p]
    if e.feature_future is not None and nf > 0:
        full[:nf, t_ctx:] = e.feature_future.to(torch.float32).T  # [p, nf] -> [nf, p]
    return (full, nf, nt)


class GiftEvalTestOverfitLoader:
    """Cycles GIFT-Eval test-window contexts as training Items (rank-sharded)."""

    def __init__(self, items: List[Item], *, rank: int = 0, world_size: int = 1,
                 cycle: bool = True) -> None:
        self.items = list(items)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.cycle = bool(cycle)

    @classmethod
    def from_eval_items(cls, eval_items: Iterable[EvalItem], *, rank: int = 0,
                        world_size: int = 1, cycle: bool = True,
                        min_context: int = 2) -> "GiftEvalTestOverfitLoader":
        """Merge each :class:`EvalItem` into one continuous training series
        (:func:`merge_context_horizon`) and validate at the loader boundary.

        Series shorter than ``min_context`` raw steps are **dropped**: the only hard
        floor is the window sampler's ``t_raw >= 2`` (it needs ≥1 context + ≥1
        prediction step). Short series are otherwise kept — the sampler handles
        ``t_raw < out_patch`` with an incomplete output patch (e.g. a 4-step series →
        2 ctx + 2 pred, ignoring the unused tail of the 16-wide patch), so the model
        learns short/1-step horizons too. After merging, a series that was too short
        as a bare context (G3.1: 16/657) is usually viable; any residual drop (merged
        length < 2) is logged."""
        items: List[Item] = []
        dropped = 0
        for e in eval_items:
            item = merge_context_horizon(e)
            validate_item(item)
            if item[0].shape[1] < min_context:
                dropped += 1
                continue
            items.append(item)
        if dropped:
            log.warning("GiftEvalTestOverfitLoader: dropped %d/%d series shorter than "
                        "min_context=%d (can't crop ctx+horizon; still scored by eval)",
                        dropped, dropped + len(items), min_context)
        return cls(items, rank=rank, world_size=world_size, cycle=cycle)

    @classmethod
    def from_cfg(cls, cfg, *, rank: int = 0, world_size: int = 1) -> "GiftEvalTestOverfitLoader":
        """Materialize the real GIFT-Eval test windows as merged ctx+horizon series
        (lazy/network — needs the ``gift_eval`` extras + a populated
        ``cfg.data.local_dir`` / ``$GIFT_EVAL``)."""
        from .gifteval_download import iter_eval_items

        eval_items = iter_eval_items(
            cfg.data.local_dir or "",
            terms=tuple(cfg.data.terms),
            items_per_config=cfg.eval.items_per_config,
        )
        return cls.from_eval_items(eval_items, rank=rank, world_size=world_size)

    def _shard(self) -> List[Item]:
        return self.items[self.rank :: self.world_size]

    def __iter__(self):
        shard = self._shard()
        if not shard:
            raise ValueError(
                f"GiftEvalTestOverfitLoader rank={self.rank}/{self.world_size} has no "
                f"items (materialized {len(self.items)} contexts; check the download / "
                f"eval.items_per_config / world_size)."
            )
        while True:
            for item in shard:
                yield item
            if not self.cycle:
                return

    def __len__(self) -> int:
        return len(self._shard())
