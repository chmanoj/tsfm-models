"""Stand-in pretrain loader (§5.2).

Mirrors the **GiftEvalPretrain** corpus, which is *not* downloadable, so it stays
synthetic. Yields exactly the frozen contract — ``(data_tensor: float32 [n, t],
num_features, num_targets)``, features-first, raw, NaNs allowed, with ``n`` and
``t`` varying per item. Nothing downstream of the loader changes when the real
Pretrain loader drops in behind ``build_loader``.

Series are sharded disjointly by rank (§9 distributed seam; no-op at
world_size=1). Reproducible: each series is generated from a seed derived from
``(global_seed, series_index)``.
"""

from __future__ import annotations

import numpy as np
import torch

from . import synthetic as S
from .contract import validate_item

# Synthetic mixture kinds (mirror D13 / cfg.data.synthetic_mix).
_KINDS = ("shared_factor", "univariate", "lag_probe")


class StandInPretrainLoader:
    def __init__(
        self,
        *,
        n_series: int,
        C_distribution=(1, 21),
        length_distribution=(64, 100000),
        nan_cap: float = 0.3,
        synthetic_mix=None,
        nan_series_prob: float = 0.5,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.n_series = int(n_series)
        self.c_lo, self.c_hi = int(C_distribution[0]), int(C_distribution[1])
        self.len_lo, self.len_hi = int(length_distribution[0]), int(length_distribution[1])
        self.nan_cap = float(nan_cap)
        self.nan_series_prob = float(nan_series_prob)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

        mix = dict(synthetic_mix or {"shared_factor": 0.5, "univariate": 0.4, "lag_probe": 0.1})
        weights = np.array([mix.get(k, 0.0) for k in _KINDS], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            raise ValueError(f"synthetic_mix must have positive total weight; got {mix}")
        self.mix_p = weights / total

    @classmethod
    def from_cfg(cls, cfg, *, rank: int = 0, world_size: int = 1) -> "StandInPretrainLoader":
        d = cfg.data
        return cls(
            n_series=d.n_series,
            C_distribution=d.C_distribution,
            length_distribution=d.length_distribution,
            nan_cap=d.nan_cap,
            synthetic_mix=dict(d.synthetic_mix),
            seed=cfg.run.seed,
            rank=rank,
            world_size=world_size,
        )

    def _make(self, idx: int):
        # Per-series RNG: disjoint, reproducible, independent of shard layout.
        rng = np.random.default_rng((self.seed, idx))
        kind = _KINDS[rng.choice(len(_KINDS), p=self.mix_p)]
        n = int(rng.integers(self.len_lo, self.len_hi + 1))

        if kind == "univariate":
            x = S.gen_primitive(rng, n)[None, :]
            nf, nt = 0, 1
        elif kind == "lag_probe":
            arr, _lag = S.gen_lag_probe(rng, n)
            x, nf, nt = arr, 1, 1
        else:  # shared_factor (all-target by default, D13)
            c_hi = max(2, self.c_hi)
            C = int(rng.integers(max(2, self.c_lo), c_hi + 1))
            x = S.gen_shared_factor(rng, n, C)
            nf, nt = 0, C

        if self.nan_cap > 0 and rng.random() < self.nan_series_prob:
            x = S.inject_nans(rng, x, self.nan_cap)

        data = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        item = (data, int(nf), int(nt))
        validate_item(item)
        return item

    def __iter__(self):
        # Disjoint round-robin shard by rank (§9). Series indices, hence content,
        # never overlap across ranks.
        for idx in range(self.rank, self.n_series, self.world_size):
            yield self._make(idx)

    def __len__(self) -> int:
        return len(range(self.rank, self.n_series, self.world_size))
