"""Streaming packer — reservoir + best-fit-decreasing (D9.3, S11).

A :class:`StreamingReservoir` wraps a base loader and turns its item stream into a
stream of pack-ready **steps**. Per step it:

1. tops a reservoir of ``K`` ``(item, SegmentSpec)`` entries up from the loader
   (``window_sampler`` samples one crop per item; the reservoir doubles as a
   shuffle buffer);
2. drains the reservoir by **best-fit-decreasing** — open a buffer and repeatedly
   place the largest spec that still fits the residual ``L`` until nothing fits or
   the residual drops below ``tail_tolerance·L`` (low single-digit tail waste).
   Every spec satisfies ``S ≤ L`` by the window-sampler budget, so giants simply
   take a buffer alone;
3. cost-buckets the formed buffers into similar-cost ``B``-buffer steps
   (:mod:`scheduler`, D9.4);
4. ``assemble()``\\ s only the emitted buffers' specs and yields one step as
   ``list[list[AssembledSegment]]``.

The frozen :func:`pack` is **not** called here — it lives in the
:func:`packed_batches` adapter between this reservoir and the train loop, so the
reservoir stays collator-free and the loop only iterates. Per-rank and free of
global state (O6): the shuffle seed is rank-offset and the base loader is sharded
disjointly by ``build_loader``; the cross-rank cost all-gather is S13.

State (D13, minimal-but-real): :meth:`state_dict` / :meth:`load_state_dict` capture
the RNG, the loader cursor (items pulled), the pending reservoir specs, and any
already-formed-but-unyielded steps, so a run resumes exactly at the same world
size. Full re-shard at a *different* world size is S13.
"""

from __future__ import annotations

import copy
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..data.contract import Item, build_loader
from ..tokenize.assemble import AssembledSegment, assemble
from ..tokenize.spec import SegmentSpec
from ..tokenize.window_sampler import SamplerParams, sample_window
from . import scheduler as SCHED
from .collator import Batch, pack

# One reservoir entry: the source item and the crop spec sampled from it. The
# item rides along so the chosen specs can be assembled when their buffer is
# emitted (assembly is deferred to emission, keeping pending state lightweight).
Entry = Tuple[Item, SegmentSpec]
Buffer = List[Entry]


class StreamingReservoir:
    """Reservoir + best-fit-decreasing + cost-bucketing over a base loader (D9.3)."""

    def __init__(
        self,
        loader,
        params: SamplerParams,
        *,
        l_pack: int,
        p_out: int,
        buffers_per_step: int,
        reservoir_k: int = 1000,
        scheduler_window: int = 128,
        tail_tolerance: float = 0.05,
        seed: int = 0,
        rank: int = 0,
    ) -> None:
        self.loader = loader
        self.params = params
        self.l_pack = int(l_pack)
        self.p_out = int(p_out)
        self.B = int(buffers_per_step)
        self.k = int(reservoir_k)
        self.window = int(scheduler_window)
        self.tail_tol = float(tail_tolerance)
        self.rank = int(rank)
        # Rank-offset shuffle seed (O6): disjoint shards already come from the
        # rank-sharded loader; this just decorrelates crop sampling across ranks.
        self._rng = np.random.default_rng((int(seed), self.rank))

        self._reservoir: List[Entry] = []        # pending (item, spec)
        self._pending_steps: List[List[Buffer]] = []  # formed, not-yet-yielded
        self._items_pulled: int = 0              # loader cursor (for resume)
        self._base_iter: Optional[Iterator[Item]] = None
        self._exhausted = False

    @classmethod
    def from_cfg(cls, cfg, *, rank: int = 0, world_size: int = 1) -> "StreamingReservoir":
        loader = build_loader(cfg, rank=rank, world_size=world_size)
        params = SamplerParams(
            l_pack=cfg.packing.L_pack,
            p_out=cfg.model.out_patch,
            tier_prior=tuple(cfg.model.tier_alloc_per_channel),
            kff_reveal_prob=float(getattr(cfg.data, "kff_reveal_prob", 0.0)),
            max_query_tokens=cfg.packing.max_query_tokens,
        )
        return cls(
            loader,
            params,
            l_pack=cfg.packing.L_pack,
            p_out=cfg.model.out_patch,
            buffers_per_step=cfg.packing.buffers_per_step,
            reservoir_k=cfg.packing.reservoir_k,
            scheduler_window=cfg.packing.scheduler_window,
            tail_tolerance=cfg.packing.tail_tolerance,
            seed=cfg.run.seed,
            rank=rank,
        )

    # --- iteration -------------------------------------------------------------

    def __iter__(self) -> "StreamingReservoir":
        return self

    def __next__(self) -> List[List[AssembledSegment]]:
        if not self._pending_steps:
            self._form_round()
        if not self._pending_steps:
            raise StopIteration
        step = self._pending_steps.pop(0)
        # Assemble only the emitted buffers (D9.3: CPU workers assemble chosen specs).
        return [[assemble(item, spec, self.p_out) for (item, spec) in buf] for buf in step]

    def _ensure_started(self) -> None:
        if self._base_iter is not None:
            return
        self._base_iter = iter(self.loader)
        # Resume: skip items already captured in the saved reservoir/pending state.
        for _ in range(self._items_pulled):
            try:
                next(self._base_iter)
            except StopIteration:
                self._exhausted = True
                break

    def _topup_reservoir(self) -> None:
        """Pull items until the reservoir holds ``K`` entries (or loader drained)."""
        self._ensure_started()
        while len(self._reservoir) < self.k and not self._exhausted:
            try:
                item = next(self._base_iter)
            except StopIteration:
                self._exhausted = True
                break
            self._items_pulled += 1
            data, nf, nt = item
            spec = sample_window(nf, nt, int(data.shape[1]), self.params, self._rng)
            self._reservoir.append((item, spec))

    def _form_one_buffer(self) -> Buffer:
        """Best-fit-decreasing: largest spec that fits, until residual < tol·L."""
        buf: Buffer = []
        residual = self.l_pack
        floor = self.tail_tol * self.l_pack
        res = self._reservoir
        while res:
            best_i, best_s = -1, -1
            for i, (_item, spec) in enumerate(res):
                s = spec.S
                if s <= residual and s > best_s:
                    best_i, best_s = i, s
            if best_i < 0:
                break
            buf.append(res.pop(best_i))
            residual -= best_s
            if residual < floor:
                break
        return buf

    def _form_round(self) -> None:
        """Fill a scheduler window of buffers, cost-bucket into B-buffer steps."""
        buffers: List[Buffer] = []
        while len(buffers) < self.window:
            self._topup_reservoir()
            if not self._reservoir:
                break
            buf = self._form_one_buffer()
            if not buf:
                break
            buffers.append(buf)
        if not buffers:
            return
        self._pending_steps = SCHED.cost_bucketed_steps(
            buffers,
            buffers_per_step=self.B,
            cost_of=lambda b: SCHED.buffer_cost(spec for (_item, spec) in b),
        )

    # --- checkpointing (minimal-but-real; full re-shard is S13) ----------------

    def state_dict(self) -> dict:
        """Capture resumable state: RNG, loader cursor, reservoir, pending steps."""
        return copy.deepcopy(
            {
                "rng_state": self._rng.bit_generator.state,
                "items_pulled": self._items_pulled,
                "reservoir": list(self._reservoir),
                "pending_steps": list(self._pending_steps),
                "exhausted": self._exhausted,
            }
        )

    def load_state_dict(self, state: dict) -> None:
        state = copy.deepcopy(state)
        self._rng.bit_generator.state = state["rng_state"]
        self._items_pulled = int(state["items_pulled"])
        self._reservoir = list(state["reservoir"])
        self._pending_steps = list(state["pending_steps"])
        self._exhausted = bool(state["exhausted"])
        # Re-derive the loader cursor lazily, skipping the already-pulled items.
        self._base_iter = None


def packed_batches(
    groups,
    *,
    l_pack: int,
    p_out: int,
    num_buffers: int,
) -> Iterator[Batch]:
    """Collation adapter: materialize each reservoir step into a :class:`Batch`.

    This is where the **frozen** :func:`pack` is called — between the reservoir and
    the train loop, so the loop only iterates ``Batch`` objects and the reservoir
    stays collator-free. ``num_buffers`` pins the static ``B`` (short final steps
    are pad-filled by ``pack``).
    """
    for group in groups:
        yield pack(group, l_pack=l_pack, p_out=p_out, num_buffers=num_buffers)
