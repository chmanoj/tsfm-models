"""Stateless ``pack()`` collator (walkthrough Stage 4, S6) → the ``Batch``.

``pack`` is **pure tensor materialization**: it takes a *caller-provided grouping*
(``list[list[AssembledSegment]]`` — one inner list per buffer) and lays each
segment channel-major into its buffer at static length ``L``, pads the tail with
``sample_id == -1``, and produces the 16-field ``Batch`` (walkthrough Stage 4 =
plan §1). It does **no** tokenization (segments arrive already assembled) and
**no** best-fit packing — buffer grouping is owned by the caller (the trivial
path in S6/S10, the reservoir in S11). The signature is frozen so both paths call
it unchanged.

Per-segment quantities are rebased into buffer-global frames:
- ``raw_start`` is offset by the segment's base in the buffer's concatenated
  ``norm_values`` store (OBSERVED tokens only; ``-1`` stays ``-1``).
- ``variate_uid`` is offset so every ``(sample, channel)`` in a buffer is unique
  (D4 needs distinct ids within a buffer; attention never crosses buffers).
- ``stats_a``/``stats_sigma`` are broadcast per token from the segment's
  per-channel ``(a, σΔ[0])`` (base scale, D10 Stage-9 inversion).

``channel_idx`` stays *sample-local* (channel within its own sample), unlike
``variate_uid``. ``R`` (the raw-store width) is the max per-buffer store length,
ragged→padded — the only quantity not bounded by ``L`` (walkthrough Stage 3).
Aux targets are not a Batch field: they are gathered from ``norm_values`` at loss
time (S8), so the collator only carries ``valid_aux``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..constants import ContentState, PAD_SAMPLE_ID, Role, RoleFT
from ..tokenize.assemble import AssembledSegment


@dataclass
class Batch:
    """One compiled step: ``B`` buffers of static length ``L`` (walkthrough Stage 4).

    All token side tensors are ``[B, L]``; the raw store is ``[B, R]``; dense
    horizon GT is ``[B, L, P_out]``. Token geometry lives entirely in these side
    tensors (D8: no buffer-index positional encoding)."""

    sample_id: torch.Tensor       # int32  [B, L]  segment id within buffer (-1 = pad)
    channel_idx: torch.Tensor     # int32  [B, L]  channel within its sample (-1 = pad)
    t_center: torch.Tensor        # float32[B, L]  continuous time vs origin (D8)
    tier_id: torch.Tensor         # int8   [B, L]  0..5; per-tier encoder & span emb
    role: torch.Tensor            # int8   [B, L]  Role {CTX, QRY} — drives the mask
    content_state: torch.Tensor   # int8   [B, L]  ContentState {OBSERVED, MASK, NA}
    role_ft: torch.Tensor         # int8   [B, L]  RoleFT {FEATURE, TARGET}
    raw_start: torch.Tensor       # int32  [B, L]  offset into norm_values (-1 unless OBSERVED)
    variate_uid: torch.Tensor     # int32  [B, L]  buffer-unique per (sample, channel) (D4)
    valid_aux: torch.Tensor       # bool   [B, L]  next-patch aux target well-defined?
    norm_values: torch.Tensor     # float32[B, R]  per-buffer normalized raw store
    observed: torch.Tensor        # bool   [B, R]  observed indicator (D7)
    stats_a: torch.Tensor         # float32[B, L]  per-token anchor (Stage-9 inversion)
    stats_sigma: torch.Tensor     # float32[B, L]  per-token base scale σΔ[0]
    horizon_target: torch.Tensor  # float32[B, L, P_out]  GT horizon (real only at QRY)
    target_valid: torch.Tensor    # bool   [B, L, P_out]  true only at QRY w/ non-NaN GT

    def to(self, device) -> "Batch":
        """Move all tensor fields to ``device`` (additive convenience for train)."""
        from dataclasses import fields
        return Batch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    @property
    def B(self) -> int:
        return self.sample_id.shape[0]

    @property
    def L(self) -> int:
        return self.sample_id.shape[1]

    @property
    def R(self) -> int:
        return self.norm_values.shape[1]


def pack(
    buffers: Sequence[Sequence[AssembledSegment]],
    *,
    l_pack: int,
    p_out: int,
    num_buffers: Optional[int] = None,
) -> Batch:
    """Materialize a ``Batch`` from a caller-provided buffer grouping.

    Args:
        buffers: one inner sequence of :class:`AssembledSegment` per buffer, placed
            in order (channel-major within each segment). ``Σ S`` per buffer must
            be ``≤ l_pack``.
        l_pack: static tokens per buffer (``L``).
        p_out: horizon patch width (``P_out``); dense GT is ``[B, L, p_out]``.
        num_buffers: static buffer count ``B``; defaults to ``len(buffers)``.
            Extra buffers beyond the grouping are all-pad. ``len(buffers) ≤ B``.
    """
    B = num_buffers if num_buffers is not None else len(buffers)
    if len(buffers) > B:
        raise ValueError(f"got {len(buffers)} buffers but num_buffers={B}")
    L = l_pack

    # --- pad-initialized side arrays (numpy at CPU pack-time, plan §2.1) ---
    sample_id = np.full((B, L), PAD_SAMPLE_ID, np.int32)
    channel_idx = np.full((B, L), -1, np.int32)
    t_center = np.zeros((B, L), np.float32)
    tier_id = np.zeros((B, L), np.int8)
    role = np.full((B, L), int(Role.CTX), np.int8)
    # Pad content_state = NA so pad tokens never route to an encoder (routing is
    # exactly content_state==OBSERVED) and get the inert learned [NA] vector.
    content_state = np.full((B, L), int(ContentState.NA), np.int8)
    role_ft = np.full((B, L), int(RoleFT.FEATURE), np.int8)
    raw_start = np.full((B, L), -1, np.int32)
    variate_uid = np.full((B, L), -1, np.int32)
    valid_aux = np.zeros((B, L), bool)
    stats_a = np.zeros((B, L), np.float32)
    stats_sigma = np.ones((B, L), np.float32)  # 1.0 keeps pad inversion finite (discarded)
    horizon_target = np.zeros((B, L, p_out), np.float32)
    target_valid = np.zeros((B, L, p_out), bool)

    store_chunks: List[List[np.ndarray]] = [[] for _ in range(B)]
    obs_chunks: List[List[np.ndarray]] = [[] for _ in range(B)]
    store_lens = np.zeros(B, np.int64)

    for bi, segs in enumerate(buffers):
        tok = 0            # token cursor within the buffer
        store = 0          # offset within the buffer's concatenated norm store
        variate_base = 0   # running channel count → buffer-unique variate ids
        for local_id, seg in enumerate(segs):
            S = seg.S
            if tok + S > L:
                raise ValueError(
                    f"buffer {bi} overflow: placing segment of size {S} at offset "
                    f"{tok} exceeds l_pack={L}"
                )
            sl = slice(tok, tok + S)
            assert seg.horizon_target.shape[1] == p_out, (seg.horizon_target.shape, p_out)

            sample_id[bi, sl] = local_id
            channel_idx[bi, sl] = seg.channel
            t_center[bi, sl] = seg.t_center
            tier_id[bi, sl] = seg.tier_id
            role[bi, sl] = seg.role
            content_state[bi, sl] = seg.content_state
            role_ft[bi, sl] = seg.role_ft
            valid_aux[bi, sl] = seg.valid_aux
            horizon_target[bi, sl] = seg.horizon_target
            target_valid[bi, sl] = seg.target_valid

            # raw_start: rebase OBSERVED windows into the buffer's store; -1 stays -1
            rs = seg.raw_start.astype(np.int32).copy()
            has_win = rs >= 0
            rs[has_win] += store
            raw_start[bi, sl] = rs

            # variate_uid: per-segment local channel id → buffer-unique id
            vu = seg.variate_uid.astype(np.int32).copy()
            has_var = vu >= 0
            vu[has_var] += variate_base
            variate_uid[bi, sl] = vu

            # per-token stats broadcast from per-channel (a, σΔ[0]); channel >= 0
            ch = seg.channel.astype(np.int64)
            stats_a[bi, sl] = seg.stats_a[ch]
            stats_sigma[bi, sl] = seg.stats_sigma_delta[ch, 0]

            store_chunks[bi].append(seg.norm_values)
            obs_chunks[bi].append(seg.observed)
            store += seg.norm_values.shape[0]
            variate_base += seg.C
            tok += S
        store_lens[bi] = store

    R = int(max(1, store_lens.max())) if B > 0 else 1
    norm_values = np.zeros((B, R), np.float32)
    observed = np.zeros((B, R), bool)
    for bi in range(B):
        if store_chunks[bi]:
            vals = np.concatenate(store_chunks[bi]).astype(np.float32)
            obs = np.concatenate(obs_chunks[bi])
            norm_values[bi, : vals.shape[0]] = vals
            observed[bi, : obs.shape[0]] = obs

    return Batch(
        sample_id=torch.from_numpy(sample_id),
        channel_idx=torch.from_numpy(channel_idx),
        t_center=torch.from_numpy(t_center),
        tier_id=torch.from_numpy(tier_id),
        role=torch.from_numpy(role),
        content_state=torch.from_numpy(content_state),
        role_ft=torch.from_numpy(role_ft),
        raw_start=torch.from_numpy(raw_start),
        variate_uid=torch.from_numpy(variate_uid),
        valid_aux=torch.from_numpy(valid_aux),
        norm_values=torch.from_numpy(norm_values),
        observed=torch.from_numpy(observed),
        stats_a=torch.from_numpy(stats_a),
        stats_sigma=torch.from_numpy(stats_sigma),
        horizon_target=torch.from_numpy(horizon_target),
        target_valid=torch.from_numpy(target_valid),
    )
