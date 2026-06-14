"""Sanity train loader + matched eval shard (simple-synthetic bring-up).

``SanityTrainLoader`` streams the **context portion** of the sanity pool as frozen
:class:`~tetris.data.contract.Item`\\ s — the held-out horizon is never shown to
the model. It **cycles** over a small ``n_series`` pool (overfit sanity) and shards
the pool disjointly by ``(rank, world_size)`` (O6, like ``StandInPretrainLoader``)
so DDP/checkpoint-reshard keep working. ``make_sanity_eval_shard`` builds the
matched :class:`~tetris.data.contract.EvalItem`\\ s — *same* series, last
``horizon`` steps held out as ``y_true`` with the dataset-provided ``season_length``
for MASE.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from .contract import EvalItem, validate_item
from .sanity import SanitySpec


class SanityTrainLoader:
    """Cycles the sanity pool's context windows as training Items (rank-sharded)."""

    def __init__(self, spec: SanitySpec, *, rank: int = 0, world_size: int = 1,
                 cycle: bool = True) -> None:
        self.spec = spec
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.cycle = bool(cycle)

    @classmethod
    def from_cfg(cls, cfg, *, rank: int = 0, world_size: int = 1) -> "SanityTrainLoader":
        return cls(SanitySpec.from_cfg(cfg), rank=rank, world_size=world_size)

    def _context_item(self, idx: int):
        data, nf, nt, _m = self.spec.make(idx)
        ctx = data[:, : self.spec.context_len]  # hold out the last `horizon` steps
        item = (torch.from_numpy(np.ascontiguousarray(ctx, dtype=np.float32)), nf, nt)
        validate_item(item)
        return item

    def __iter__(self):
        # Disjoint round-robin shard by rank (O6); cycle for an effectively
        # unbounded overfit stream (run stops at `steps`).
        shard = range(self.rank, self.spec.n_series, self.world_size)
        while True:
            for idx in shard:
                yield self._context_item(idx)
            if not self.cycle:
                return

    def __len__(self) -> int:
        return len(range(self.rank, self.spec.n_series, self.world_size))


def make_sanity_eval_shard(cfg) -> List[EvalItem]:
    """Matched eval shard: the *same* sanity series, last ``horizon`` steps held
    out as ``y_true`` with the dataset-provided ``season_length`` (for MASE)."""
    spec = SanitySpec.from_cfg(cfg)
    p = spec.horizon
    items: List[EvalItem] = []
    for idx in range(spec.n_series):
        data, nf, nt, m = spec.make(idx)
        context = torch.from_numpy(np.ascontiguousarray(data[:, :-p], dtype=np.float32))
        y_true = torch.from_numpy(np.ascontiguousarray(data[nf:, -p:].T, dtype=np.float32))
        items.append(EvalItem(
            data_tensor=context, num_features=nf, num_targets=nt,
            y_true=y_true, naive_denom=None,
            config_id=f"sanity/{spec.case}/m{m}/series_{idx}", season_length=m,
        ))
    return items
