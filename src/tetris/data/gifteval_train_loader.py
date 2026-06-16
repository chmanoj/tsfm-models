"""GIFT-Eval ``train``-split training loader (G5 — in-distribution gold).

The GIFT-Eval ``train`` split (everything *before* the test/validation windows,
``Dataset.training_dataset``) is legal in-distribution training data and is used by
the current leaderboard models (Moirai 2.0 precedent; decision-log D13 corpus
component **B**). This loader is the **live** factory key for it — it wraps
:func:`~tetris.data.gifteval_download.iter_train_items` directly (no on-disk
materialization), so the real ``train`` series stream through ``build_loader``
exactly like any other corpus, yielding the frozen ``Item`` 3-tuple.

It differs from the G3/G3.1 ``gifteval_test_overfit`` loader: that one merges the
held-out *test* horizon back into the series (a deliberate memorization probe,
**not** zero-shot); this one reads the genuine train split and never touches the
test data — so a curriculum that blends it in stays honest.

Series are rank-sharded disjointly by ``iter_train_items`` itself (round-robin over
the flattened ``(config, series)`` stream, O6; no-op at ``world_size=1``) and the
stream is **cycled** (the GIFT-Eval train split is finite; the run stops at
``steps``). Lazy/network: needs the ``gift_eval`` extras + a populated
``cfg.data.local_dir`` / ``$GIFT_EVAL`` — never imported in CI.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

from .contract import Item

log = logging.getLogger("tetris.gifteval")


class GiftEvalTrainLoader:
    """Stream the GIFT-Eval ``train`` split as frozen training ``Item``\\ s (O6).

    Thin, lazy wrapper over :func:`iter_train_items`: it owns the rank/world_size
    sharding (delegated to the iterator so the global round-robin matches every
    other loader), the term selection, and the per-config cap, and cycles the
    finite split for unbounded training.
    """

    def __init__(self, *, local_dir: str = "", term: str = "short",
                 configs: Optional[List[str]] = None,
                 max_series_per_config: int = -1, cycle: bool = True,
                 rank: int = 0, world_size: int = 1) -> None:
        self.local_dir = str(local_dir)
        self.term = str(term)
        self.configs = list(configs) if configs is not None else None
        self.max_series_per_config = int(max_series_per_config)
        self.cycle = bool(cycle)
        self.rank = int(rank)
        self.world_size = int(world_size)

    @classmethod
    def from_cfg(cls, cfg, *, rank: int = 0, world_size: int = 1) -> "GiftEvalTrainLoader":
        """Build from config. The train ``term`` defaults to the **first** of
        ``cfg.data.terms`` (``short`` gives the largest train split — its test
        carve-out is smallest); ``cfg.data.train_term`` overrides. The per-config
        series cap is ``cfg.data.train_max_series_per_config`` (``-1`` -> all)."""
        d = cfg.data
        term = d.train_term or (d.terms[0] if d.terms else "short")
        return cls(
            local_dir=d.local_dir or "",
            term=term,
            max_series_per_config=int(getattr(d, "train_max_series_per_config", -1)),
            rank=rank,
            world_size=world_size,
        )

    def __iter__(self) -> Iterator[Item]:
        from .gifteval_download import iter_train_items

        while True:
            n = 0
            for item in iter_train_items(
                self.local_dir,
                configs=self.configs,
                term=self.term,
                max_series_per_config=self.max_series_per_config,
                rank=self.rank,
                world_size=self.world_size,
            ):
                n += 1
                yield item
            if n == 0:
                raise ValueError(
                    f"GiftEvalTrainLoader rank={self.rank}/{self.world_size} yielded no "
                    f"train series (term={self.term!r}); check the download / "
                    f"cfg.data.local_dir / world_size."
                )
            if not self.cycle:
                return
