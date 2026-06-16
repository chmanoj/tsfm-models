"""G5 curriculum loader — progress-conditioned weighted source mixture (D13).

The decision-log **two-phase schedule** (D13) as a ``build_loader`` factory key:
a weighted mixture over independent component loaders (synthetic / real pretrain /
GIFT-Eval ``train`` split), where the weights **anneal with training progress**.

* **Phase 1 (broad pretrain, A+C+D with B natural):** the configured
  ``multiplier_phase1`` weights hold until ``phase2_start`` (a fraction of
  ``total_items``).
* **Phase 2 (anneal toward the train split):** over ``[phase2_start, 1]`` each
  source's multiplier interpolates linearly to ``multiplier_phase2`` — typically
  upweighting B (the train split). The crop-marginal half of D13 phase 2
  (``auto_from_test_configs``) is the reservoir's job (a progress-keyed crop
  schedule), not this loader's.

Mixture weight per source = ``(multiplier × max(1, size)^alpha) ** (1/temperature)``,
renormalized over sources with a positive multiplier (D13: manual multiplier ×
size^α; ``temperature`` > 1 flattens the mixture so rare sources/frequencies aren't
starved). ``multiplier`` 0 removes a source in that phase and its mass redistributes
automatically.

**Progress** is the loader's own yield counter over ``total_items`` — it advances in
lockstep with the reservoir's ``items_pulled`` cursor (the reservoir pulls exactly
one item per yield), so this loader and the reservoir's crop schedule see the same
progress without a shared callback. **Sharding (O6):** every component loader is
itself rank-sharded, so each rank draws from its own disjoint shard of every source;
the per-rank mixture RNG is seed+rank offset for decorrelated source draws.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .contract import Item

log = logging.getLogger("tetris.curriculum")


class CurriculumLoader:
    """Weighted, progress-annealed mixture over component loaders (D13)."""

    def __init__(self, components: List["_Source"], *, total_items: int,
                 phase2_start: float, alpha: float, temperature: float,
                 seed: int = 0, rank: int = 0, world_size: int = 1,
                 log_every_pulls: int = 0) -> None:
        if not components:
            raise ValueError("CurriculumLoader needs >=1 component source")
        self.components = components
        self.total_items = int(total_items)
        self.phase2_start = float(phase2_start)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.rank = int(rank)
        self.world_size = int(world_size)
        # Periodic INFO heartbeat (rank 0 only): progress + live weights + the source
        # histogram since the last beat — proves the phase-1→phase-2 mix shift in the
        # train log. 0 = off (the default, so tests are silent).
        self.log_every_pulls = int(log_every_pulls)
        # Per-rank RNG so source draws decorrelate across ranks (the component
        # loaders are already rank-sharded for disjoint *content*).
        self._rng = np.random.default_rng((int(seed), int(rank)))

    @classmethod
    def from_cfg(cls, cfg, *, rank: int = 0, world_size: int = 1) -> "CurriculumLoader":
        from .contract import build_loader

        c = cfg.curriculum
        components: List[_Source] = []
        for sc in c.sources:
            sub = copy.deepcopy(cfg)
            sub.data.loader = sc.loader
            if sc.shard_root:
                sub.data.shard_root = sc.shard_root
            loader = build_loader(sub, rank=rank, world_size=world_size)
            size = int(sc.n_series) if sc.n_series > 0 else _infer_size(sc)
            components.append(_Source(
                name=sc.name, loader=loader, size=max(1, size),
                mult1=float(sc.multiplier_phase1), mult2=float(sc.multiplier_phase2)))
        return cls(components, total_items=c.total_items, phase2_start=c.phase2_start,
                   alpha=c.alpha, temperature=c.temperature,
                   seed=cfg.run.seed, rank=rank, world_size=world_size,
                   log_every_pulls=(2000 if rank == 0 else 0))

    # --- schedule --------------------------------------------------------------

    def _anneal_frac(self, progress: float) -> float:
        """0 in phase 1, ramping 0→1 linearly across ``[phase2_start, 1]``."""
        if progress < self.phase2_start:
            return 0.0
        span = max(1e-9, 1.0 - self.phase2_start)
        return float(min(1.0, (progress - self.phase2_start) / span))

    def weights_at(self, progress: float) -> np.ndarray:
        """Renormalized mixture weights at a given training progress in [0, 1]."""
        f = self._anneal_frac(progress)
        eff = np.empty(len(self.components), dtype=np.float64)
        for i, s in enumerate(self.components):
            mult = s.mult1 + f * (s.mult2 - s.mult1)
            if mult <= 0:
                eff[i] = 0.0
            else:
                eff[i] = (mult * (s.size ** self.alpha)) ** (1.0 / self.temperature)
        total = eff.sum()
        if total <= 0:  # all-zero (guarded by config validation) — fall back to uniform
            eff[:] = 1.0
            total = eff.sum()
        return eff / total

    # --- iteration -------------------------------------------------------------

    def __iter__(self):
        iters = [iter(s.loader) for s in self.components]
        pulled = 0
        denom = max(1, self.total_items)
        hist = np.zeros(len(iters), dtype=np.int64)   # source draws since last heartbeat
        while True:
            progress = min(1.0, pulled / denom)
            w = self.weights_at(progress)
            j = int(self._rng.choice(len(iters), p=w))
            try:
                item: Item = next(iters[j])
            except StopIteration:  # a finite (cycle=False) component — refresh it
                iters[j] = iter(self.components[j].loader)
                item = next(iters[j])
            pulled += 1
            hist[j] += 1
            if self.log_every_pulls and pulled % self.log_every_pulls == 0:
                phase = "phase1" if progress < self.phase2_start else "phase2/anneal"
                mix = " ".join(f"{s.name}:w={w[i]:.2f},n={int(hist[i])}"
                               for i, s in enumerate(self.components))
                log.info("curriculum pull=%d progress=%.3f %s | %s",
                         pulled, progress, phase, mix)
                hist[:] = 0
            yield item


class _Source:
    """A built component loader + its mixture metadata."""

    __slots__ = ("name", "loader", "size", "mult1", "mult2")

    def __init__(self, name, loader, *, size, mult1, mult2) -> None:
        self.name = name
        self.loader = loader
        self.size = int(size)
        self.mult1 = float(mult1)
        self.mult2 = float(mult2)


def _infer_size(sc) -> int:
    """Best-effort size for the ``size^α`` term: a streaming corpus' manifest
    ``n_series`` (read cheaply, no shard touch), else 1 (the live train split has no
    cheap size; set ``SourceCfg.n_series`` to weight it explicitly)."""
    if sc.loader == "streaming" and sc.shard_root:
        manifest = Path(sc.shard_root) / "manifest.json"
        try:
            return int(json.loads(manifest.read_text()).get("n_series", 1))
        except Exception:  # pragma: no cover - missing/corrupt manifest
            return 1
    return 1
