"""Attention-mask construction — the D9.1 truth table (S5).

Two constructors for the same boolean reachability rule, one per backend (D14):

- **Flex path (CUDA):** a ``mask_mod(b, h, q_idx, kv_idx)`` closure handed to
  ``create_block_mask`` → ``BlockMask`` (``score_mod`` is a separate, deferred
  A3/NSA socket — not built here).
- **SDPA path (Mac/CPU):** a materialized boolean tensor ``[B, 1, L, L]``
  (``True`` = the (query, key) pair participates), head-broadcast.

Both encode the **same** D9 rule and are equality-tested (``test_masks``). The
mask is **depth-invariant in v1** (D14): built once per batch, reused by every
layer. Attention takes it as a parameter, so a future per-layer schedule (the
A3 ladder) is an interface addition, not a refactor.

D9 truth table — q attends k iff (role ∈ {CTX, QRY}; KFF is CTX, so it falls out
for free: queries read it, context can't, it never reads queries)::

    sample_id[q] == sample_id[k] AND
      ( (role[k]==CTX AND (role[q]==QRY OR t_center[k] <= t_center[q]))
        OR (role[k]==QRY AND role[q]==QRY) )

- ctx→ctx: time-causal, **channel-blind** (``<=`` includes same-time and self).
- qry→ctx: always.   · qry→qry: always (bidirectional trajectory coherence).
- ctx→qry: never (context reps independent of any attached horizon).

Attention **never crosses buffers or samples** — the ``sample_id`` equality
enforces both boundaries (pad has ``sample_id == -1``, distinct from every real
id). Pad positions get a ``q == k`` self-attend exception (NaN guard; their
outputs are discarded), so every row has ≥1 key and softmax never sees an
all-masked row. (D8: token geometry comes only from these side tensors.)
"""

from __future__ import annotations

from typing import Callable

import torch

from .constants import PAD_SAMPLE_ID, Role


def build_sdpa_mask(
    sample_id: torch.Tensor,
    role: torch.Tensor,
    t_center: torch.Tensor,
) -> torch.Tensor:
    """Materialized SDPA boolean mask ``[B, 1, L, L]`` (the Mac/CPU path).

    Inputs are the per-token side tensors ``[B, L]`` (``sample_id`` int,
    ``role`` int in :class:`Role`, ``t_center`` float). Returns ``True`` where
    query ``q`` may attend key ``k``; the singleton head dim broadcasts over
    ``H`` in :func:`tetris.backend.attend`.
    """
    s_q = sample_id.unsqueeze(2)          # [B, L, 1]  query along dim 1
    s_k = sample_id.unsqueeze(1)          # [B, 1, L]  key along dim 2
    rq = role.unsqueeze(2)
    rk = role.unsqueeze(1)
    tq = t_center.unsqueeze(2)
    tk = t_center.unsqueeze(1)

    same = s_q == s_k                                         # same sample & buffer
    q_is_qry = rq == int(Role.QRY)
    k_is_ctx = rk == int(Role.CTX)
    k_is_qry = rk == int(Role.QRY)

    ctx_rule = k_is_ctx & (q_is_qry | (tk <= tq))            # qry→ctx always; ctx→ctx causal
    qry_rule = k_is_qry & q_is_qry                           # qry→qry bidirectional
    core = same & (ctx_rule | qry_rule)                      # [B, L, L]

    # Pad rows (sample_id == -1) attend only themselves (NaN guard). Because pad
    # ids never equal a real id, `same` already masks pad keys for real queries
    # and real keys for pad queries; this just restores the diagonal for pad q.
    L = sample_id.shape[1]
    eye = torch.eye(L, dtype=torch.bool, device=sample_id.device).unsqueeze(0)
    q_is_pad = (s_q == PAD_SAMPLE_ID)                        # [B, L, 1]
    allow = torch.where(q_is_pad, eye, core)

    return allow.unsqueeze(1)                                # [B, 1, L, L]


def make_mask_mod(
    sample_id: torch.Tensor,
    role: torch.Tensor,
    t_center: torch.Tensor,
) -> Callable:
    """Build the FlexAttention ``mask_mod`` closure for the D9 rule (CUDA path).

    The returned callable ``(b, h, q_idx, kv_idx) -> bool`` captures the side
    tensors and is the exact object handed to ``create_block_mask``. It is the
    same boolean formula as :func:`build_sdpa_mask`, written for (broadcastable)
    index tensors — so it also densifies by enumeration for the equality test.
    """
    ctx = int(Role.CTX)
    qry = int(Role.QRY)

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001 (h unused: head-invariant)
        s_q = sample_id[b, q_idx]
        s_k = sample_id[b, kv_idx]
        rq = role[b, q_idx]
        rk = role[b, kv_idx]
        tq = t_center[b, q_idx]
        tk = t_center[b, kv_idx]

        same = s_q == s_k
        q_is_qry = rq == qry
        ctx_rule = (rk == ctx) & (q_is_qry | (tk <= tq))
        qry_rule = (rk == qry) & q_is_qry
        core = same & (ctx_rule | qry_rule)

        q_is_pad = s_q == PAD_SAMPLE_ID
        return torch.where(q_is_pad, q_idx == kv_idx, core)

    return mask_mod


def build_block_mask(
    sample_id: torch.Tensor,
    role: torch.Tensor,
    t_center: torch.Tensor,
):
    """FlexAttention ``BlockMask`` for the D9 rule (the CUDA path).

    Head dim is broadcast (``H=None``). FlexAttention is lazily imported because
    it is only the live attention kernel on CUDA; on CPU this still *builds*
    (block-granular), used to confirm the constructor is wired, while exact
    equality is checked elementwise against :func:`build_sdpa_mask` via the
    shared :func:`make_mask_mod`.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    B, L = sample_id.shape
    mask_mod = make_mask_mod(sample_id, role, t_center)
    return create_block_mask(
        mask_mod, B=B, H=None, Q_LEN=L, KV_LEN=L, device=str(sample_id.device)
    )


def dense_from_mask_mod(mask_mod: Callable, B: int, L: int, device=None) -> torch.Tensor:
    """Enumerate a ``mask_mod`` to a dense boolean ``[B, L, L]`` (test/util).

    Evaluates the closure on the full index grid (element-level, not the
    block-granular ``BlockMask.to_dense``), so it exactly reflects what
    FlexAttention applies per element on CUDA.
    """
    b = torch.arange(B, device=device).view(B, 1, 1)
    q = torch.arange(L, device=device).view(1, L, 1)
    k = torch.arange(L, device=device).view(1, 1, L)
    out = mask_mod(b, None, q, k)
    return out.expand(B, L, L).clone()
