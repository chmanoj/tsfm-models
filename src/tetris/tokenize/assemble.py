"""assemble (walkthrough Stages 2–3 boundary) — the pure per-segment tokenizer.

Given ``(item, spec, p_out)`` it: normalizes each channel (D10, context-only),
builds a **flat per-segment normalized raw store** (``norm_values``/``observed``,
indexed by each token's ``raw_start`` — the gather store), and emits the per-token
side tensors. Horizon ground-truth is emitted **dense over the segment**
(``[S, P_out]`` + valid), matching the Batch's ``[B,L,P_out]`` (no per-batch Qmax).
Pure — no buffer/global state (S9).

Token tagging uses the three decision-log enums:
- ``role ∈ {CTX, QRY}`` (D6/D9) — drives the attention mask.
- ``content_state ∈ {OBSERVED, MASK, NA}`` (D7) — selects the content slot.
- ``role_ft ∈ {FEATURE, TARGET}`` (D11) — the added role embedding.

A **KFF token** is just an observed CTX feature token at ``t_center > 0`` (D11) —
``role=CTX, content_state=OBSERVED, role_ft=FEATURE``, encoded by the P_out tier
encoder; no KFF role/state exists. ``norm_values`` uses the base scale
``sigma_delta[0]`` (D10 input z); per-tier ``sigma_delta[1..5]`` ride along for
aux targets / receipts. Encoder-routed tokens are exactly ``content_state==OBSERVED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .. import normalize as norm
from .. import telescope as TS
from ..constants import ChannelRole, ContentState, PATCH, Role, RoleFT, V
from .spec import SegmentSpec

_NO_RAW = -1  # raw_start sentinel for tokens with no encoder window (MASK/NA)


@dataclass
class AssembledSegment:
    S: int
    C: int
    # per-token side arrays (channel-major), length S
    tier_id: np.ndarray        # int8   [S]
    channel: np.ndarray        # int32  [S]
    raw_start: np.ndarray      # int32  [S]  offset into this segment's norm_values
    role: np.ndarray           # int8   [S]  Role {CTX, QRY}
    content_state: np.ndarray  # int8   [S]  ContentState {OBSERVED, MASK, NA}
    role_ft: np.ndarray        # int8   [S]  RoleFT {FEATURE, TARGET}
    variate_uid: np.ndarray    # int32  [S]  per-(segment,channel) local id
    t_center: np.ndarray       # float32[S]
    valid_aux: np.ndarray      # bool   [S]
    # flat per-segment normalized raw store (concatenated per channel)
    norm_values: np.ndarray    # float32[R_seg]
    observed: np.ndarray       # bool   [R_seg]
    # per-channel stats (a + per-tier sigma); receipts for D4 + Stage-9 inversion
    stats_a: np.ndarray            # float32[C]
    stats_sigma_delta: np.ndarray  # float32[C, V]
    # dense horizon ground truth (only query slots carry real data)
    horizon_target: np.ndarray     # float32[S, P_out]
    target_valid: np.ndarray       # bool   [S, P_out]


def _norm_window(raw: np.ndarray, lo: int, length: int, a: float, sigma: float):
    """Normalized values + observed indicator for raw indices ``[lo, lo+length)``
    (out-of-range / NaN → value 0, observed False)."""
    n = raw.shape[0]
    idx = lo + np.arange(length)
    in_range = (idx >= 0) & (idx < n)
    safe = np.clip(idx, 0, n - 1)
    obs = in_range & np.isfinite(raw[safe])
    vals = np.where(obs, np.arcsinh((raw[safe] - a) / sigma), 0.0).astype(np.float32)
    return vals, obs


def assemble(item, spec: SegmentSpec, p_out: int) -> AssembledSegment:
    data, n_features, n_targets = item
    C = spec.C
    assert data.shape[0] == C
    raw = data.detach().cpu().numpy().astype(np.float64)  # [C, t]
    origin = spec.origin
    counts = spec.counts
    cov = TS.coverage(counts)
    qspan = PATCH.index(p_out)
    q_tok = spec.q_tok

    # channel-major block layout
    horizon = [spec.horizon_tokens(c) for c in range(C)]
    block_sizes = [spec.n_eff + horizon[c] for c in range(C)]
    bases = np.concatenate([[0], np.cumsum(block_sizes)])[:-1].astype(np.int64)
    S = int(sum(block_sizes))
    assert S == spec.S, (S, spec.S)

    tier_id = np.zeros(S, np.int8)
    channel = np.full(S, -1, np.int32)
    raw_start = np.full(S, _NO_RAW, np.int32)
    role = np.full(S, int(Role.CTX), np.int8)
    content_state = np.full(S, int(ContentState.OBSERVED), np.int8)
    role_ft = np.zeros(S, np.int8)
    variate_uid = np.full(S, -1, np.int32)
    t_center = np.zeros(S, np.float32)
    valid_aux = np.zeros(S, bool)
    horizon_target = np.zeros((S, p_out), np.float32)
    target_valid = np.zeros((S, p_out), bool)

    stats_a = np.zeros(C, np.float32)
    stats_sigma_delta = np.ones((C, V), np.float32)
    store_chunks: List[np.ndarray] = []
    obs_chunks: List[np.ndarray] = []
    store_base = 0  # running offset into the segment's flat store

    for c in range(C):
        rc = raw[c]
        ctx = rc[max(0, origin - cov):origin]
        st = norm.compute_stats(torch.from_numpy(ctx))
        a = float(st.a); sigma = float(st.sigma)  # sigma = sigma_delta[0], base scale
        stats_a[c] = a
        stats_sigma_delta[c] = st.sigma_delta.numpy().astype(np.float32)

        cr = spec.channel_roles[c]
        is_target = cr == int(ChannelRole.TARGET)
        is_kff = cr == int(ChannelRole.KFF)
        rft = int(RoleFT.TARGET) if is_target else int(RoleFT.FEATURE)

        # --- per-channel flat store: context [origin-cov, origin) + (KFF) horizon ---
        store_lo = origin - cov
        ctx_vals, ctx_obs = _norm_window(rc, store_lo, cov, a, sigma)
        if is_kff:
            h_len = q_tok * p_out
            h_vals, h_obs = _norm_window(rc, origin, h_len, a, sigma)
            chunk_vals = np.concatenate([ctx_vals, h_vals])
            chunk_obs = np.concatenate([ctx_obs, h_obs])
        else:
            chunk_vals, chunk_obs = ctx_vals, ctx_obs
        store_chunks.append(chunk_vals); obs_chunks.append(chunk_obs)

        base = int(bases[c])
        # --- context tokens via telescope dispatch (channel-major) ---
        disp = TS.build_dispatch(counts, origin, channel_base=base, n_per_channel=block_sizes[c])
        for k, tslice in enumerate(disp.tiers):
            for i in range(tslice.scatter_pos.shape[0]):
                pos = int(tslice.scatter_pos[i])
                rs = int(tslice.raw_start[i]); re = int(tslice.raw_end[i])
                _, obs = _norm_window(rc, rs, PATCH[k], a, sigma)
                channel[pos] = c; variate_uid[pos] = c; tier_id[pos] = k
                role[pos] = int(Role.CTX); role_ft[pos] = rft
                t_center[pos] = (rs + re) * 0.5 - origin
                if obs.any():
                    content_state[pos] = int(ContentState.OBSERVED)
                    raw_start[pos] = store_base + (rs - store_lo)
                else:
                    content_state[pos] = int(ContentState.NA)  # fully-missing patch (D7)
                # aux next-patch validity (raw-time): region [re, re+P_k) in-context & observed
                tgt_lo, tgt_hi = re, re + PATCH[k]
                if tgt_hi <= origin and tgt_lo >= 0:
                    _, tobs = _norm_window(rc, tgt_lo, PATCH[k], a, sigma)
                    valid_aux[pos] = bool(tobs.any())

        # --- horizon tokens ---
        if horizon[c] == 0:
            store_base += chunk_vals.shape[0]
            continue
        for j in range(q_tok):
            pos = base + spec.n_eff + j
            channel[pos] = c; variate_uid[pos] = c; tier_id[pos] = qspan
            role_ft[pos] = rft
            t_center[pos] = j * p_out + p_out * 0.5
            if is_target:
                # query [MASK] token — learned embedding, routed to the horizon head
                role[pos] = int(Role.QRY)
                content_state[pos] = int(ContentState.MASK)
                hv, hobs = _norm_window(rc, origin + j * p_out, p_out, a, sigma)
                within = (j * p_out + np.arange(p_out)) < spec.p
                horizon_target[pos] = hv
                target_valid[pos] = hobs & within
            else:  # KFF: observed CTX feature token at t>0, routed to the P_out encoder
                role[pos] = int(Role.CTX)
                _, hobs = _norm_window(rc, origin + j * p_out, p_out, a, sigma)
                if hobs.any():
                    content_state[pos] = int(ContentState.OBSERVED)
                    raw_start[pos] = store_base + cov + j * p_out
                else:
                    content_state[pos] = int(ContentState.NA)
        store_base += chunk_vals.shape[0]

    norm_values = (np.concatenate(store_chunks) if store_chunks
                   else np.zeros(0, np.float32)).astype(np.float32)
    observed = (np.concatenate(obs_chunks) if obs_chunks else np.zeros(0, bool))

    return AssembledSegment(
        S=S, C=C,
        tier_id=tier_id, channel=channel, raw_start=raw_start,
        role=role, content_state=content_state, role_ft=role_ft,
        variate_uid=variate_uid, t_center=t_center, valid_aux=valid_aux,
        norm_values=norm_values, observed=observed,
        stats_a=stats_a, stats_sigma_delta=stats_sigma_delta,
        horizon_target=horizon_target, target_valid=target_valid,
    )
