"""Cost-bucketed step scheduling (D9.4, S11).

A buffer's attention cost is ``Σ_i S_i² / 2`` (the number of attention pairs),
known from the segment specs alone — no tokenization needed. Per-token attention
(~``4d·S``) vs FFN (~``12d²``) reach parity near ``S ≈ 3d``; a solo 64k buffer can
cost ~40–60× a packed-small buffer at *identical* tensor shape. To stop the
straggler tax, a scheduler window of 64–256 buffers is sorted by cost and chopped
into **similar-cost** ``B``-buffer steps — giants travel together in occasional
slow steps, and every step keeps the one static compile shape (only the buffer
*contents* differ).

This module is **single-rank / in-process** (S11). The cross-rank cost all-gather
that balances steps across ranks (§9) is wired and tested at S13; the seam is the
``cost_of`` callable + the deterministic sort here, which a global schedule reuses.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, TypeVar

from ..tokenize.spec import SegmentSpec

T = TypeVar("T")


def buffer_cost(specs: Iterable[SegmentSpec]) -> float:
    """Attention-pair cost ``Σ_i S_i² / 2`` of a buffer (D9.4), from specs alone."""
    return 0.5 * sum(float(s.S) * float(s.S) for s in specs)


def cost_bucketed_steps(
    buffers: Sequence[T],
    *,
    buffers_per_step: int,
    cost_of: Callable[[T], float],
) -> List[List[T]]:
    """Sort ``buffers`` by cost and chunk into ``buffers_per_step``-sized steps.

    Adjacent buffers in the sorted order have the closest costs, so each emitted
    step groups similar-cost buffers (D9.4). The final step may be short; the
    collator pads the missing buffers (``pack(..., num_buffers=B)``). Stable: ties
    keep input order, so the schedule is deterministic given the same window.
    """
    if buffers_per_step < 1:
        raise ValueError(f"buffers_per_step must be >= 1, got {buffers_per_step}")
    order = sorted(buffers, key=cost_of)
    return [order[i : i + buffers_per_step] for i in range(0, len(order), buffers_per_step)]
