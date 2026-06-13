"""assemble (D9.2 / D7 / D10) — the pure per-segment tokenizer; the heart of the
collator. Given ``(item, spec)`` it normalizes per channel (D10), builds the
channel-major token layout (D8), per-tier raw windows (value + observed
indicator, D7), side tensors, query/KFF tokens, and per-channel norm stats.

Layout (channel-major, segment-local positions in ``[0, S)``): each channel ``c``
occupies a contiguous block ``[base_c, base_c + n_eff + horizon_c)`` —
``n_eff`` context tokens then ``horizon_c`` horizon tokens. Horizon tokens are
query [MASK] tokens for target channels (separate, learned embedding) and
observed known-future-feature (KFF) tokens for revealed features (routed through
the P_out-tier encoder like any observed token).

Per-tier dispatch is emitted as ``(positions, values, observed)`` lists at the
*actual* token counts; the collator scatters them into the static MoE index
space (``tier_idx[k]`` of length ``n_ctx_cap``). This is a pure function — no
global/buffer state — so packed and unpacked runs are identical (S9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .. import normalize as norm
from .. import telescope as TS
from ..constants import PATCH, V, ContentState, Role, RoleFT
from .spec import SegmentSpec

VAR_FEAT_DIM = 4  # D4/D10 content-summary receipts: [log σ, arcsinh a, fallback code, log1p|a|]
_FALLBACK_CODE = {"mad": 0.0, "mean_abs": 1.0, "iqr": 2.0, "unit": 3.0}


@dataclass
class AssembledSegment:
    S: int
    C: int
    # side tensors (segment-local), each length S
    channel_idx: np.ndarray      # int32
    vslot_local: np.ndarray      # int32 (one variate per channel within a segment)
    t_center: np.ndarray         # float32 (steps vs origin; <0 past, >=0 horizon)
    span: np.ndarray             # int8 (tier index)
    role: np.ndarray             # int8 (Role)
    role_ft: np.ndarray          # int8 (RoleFT)
    content_state: np.ndarray    # int8 (ContentState)
    valid: np.ndarray            # bool
    # per-tier context dispatch (lists of length V; observed tokens only)
    tier_pos: List[np.ndarray]       # int64 [count_k] local positions
    tier_values: List[np.ndarray]    # float32 [count_k, P_k] normalized z, NaN->0
    tier_observed: List[np.ndarray]  # float32 [count_k, P_k] 0/1 indicator
    # horizon query tokens (target channels)
    qry_pos: np.ndarray          # int64 [Q]
    qry_channel: np.ndarray      # int64 [Q]
    qry_hidx: np.ndarray         # int64 [Q] horizon-patch index 0..q_tok-1
    # per-channel normalization stats (for inversion + receipts)
    norm_a: np.ndarray           # float32 [C]
    norm_sigma: np.ndarray       # float32 [C]
    norm_sigma_tier: np.ndarray  # float32 [C, V]
    var_feats: np.ndarray        # float32 [C, VAR_FEAT_DIM]


def _gather_window(z: np.ndarray, raw: np.ndarray, start: int, width: int):
    """Normalized values + observed indicator for raw indices ``[start, start+width)``,
    clipped to the series and to finite raw values (out-of-range / NaN → not observed)."""
    n = z.shape[0]
    idx = start + np.arange(width)
    in_range = (idx >= 0) & (idx < n)
    safe = np.clip(idx, 0, n - 1)
    obs = in_range & np.isfinite(raw[safe])
    vals = np.where(obs, z[safe], 0.0).astype(np.float32)
    return vals, obs.astype(np.float32)


def assemble(item, spec: SegmentSpec) -> AssembledSegment:
    data, n_features, n_targets = item
    C = spec.C
    assert data.shape[0] == C
    x = data.detach().cpu().numpy().astype(np.float64)  # [C, t], may contain NaN
    t_raw = spec.t_raw
    origin = spec.origin
    counts = spec.counts
    qspan = PATCH.index(spec.p_out)
    P_out = spec.p_out

    # channel-major block layout
    horizon = [spec.horizon_tokens(c) for c in range(C)]
    block_sizes = [spec.n_eff + horizon[c] for c in range(C)]
    bases = np.concatenate([[0], np.cumsum(block_sizes)])[:-1].astype(np.int64)
    S = int(sum(block_sizes))
    assert S == spec.S, (S, spec.S)

    channel_idx = np.full(S, -1, np.int32)
    vslot_local = np.full(S, -1, np.int32)
    t_center = np.zeros(S, np.float32)
    span = np.full(S, qspan, np.int8)
    role = np.full(S, int(Role.CTX), np.int8)
    role_ft = np.zeros(S, np.int8)
    content_state = np.full(S, int(ContentState.OBSERVED), np.int8)
    valid = np.ones(S, bool)

    tier_pos: List[list] = [[] for _ in range(V)]
    tier_values: List[list] = [[] for _ in range(V)]
    tier_observed: List[list] = [[] for _ in range(V)]
    qry_pos, qry_channel, qry_hidx = [], [], []

    norm_a = np.zeros(C, np.float32)
    norm_sigma = np.ones(C, np.float32)
    norm_sigma_tier = np.ones((C, V), np.float32)
    var_feats = np.zeros((C, VAR_FEAT_DIM), np.float32)

    cov = TS.coverage(counts)
    for c in range(C):
        rc = x[c]
        # context statistics from the context window only (no leakage, D10)
        ctx = rc[max(0, origin - cov):origin]
        st = norm.compute_stats(torch.from_numpy(ctx))
        a = float(st.a); sig = float(st.sigma)
        norm_a[c] = a; norm_sigma[c] = sig
        norm_sigma_tier[c] = st.sigma_tier.numpy().astype(np.float32)
        var_feats[c] = [np.log(sig + 1e-12), np.arcsinh(a), _FALLBACK_CODE[st.fallback], np.log1p(abs(a))]
        z = np.arcsinh((rc - a) / sig)  # whole-series normalized (KFF future uses context stats)

        base = int(bases[c])
        # --- context tokens (channel-major), via telescope dispatch ---
        disp = TS.build_dispatch(counts, origin, channel_base=base, n_per_channel=block_sizes[c])
        for k, tslice in enumerate(disp.tiers):
            for i in range(tslice.scatter_pos.shape[0]):
                pos = int(tslice.scatter_pos[i])
                rs = int(tslice.raw_start[i]); re = int(tslice.raw_end[i])
                vals, obs = _gather_window(z, rc, rs, PATCH[k])
                channel_idx[pos] = c; vslot_local[pos] = c
                span[pos] = k; role[pos] = int(Role.CTX); role_ft[pos] = spec.role_ft[c]
                t_center[pos] = (rs + re) * 0.5 - origin
                content_state[pos] = int(ContentState.OBSERVED) if obs.any() else int(ContentState.NA)
                tier_pos[k].append(pos); tier_values[k].append(vals); tier_observed[k].append(obs)

        # --- horizon tokens ---
        if horizon[c] == 0:
            continue
        is_target = spec.role_ft[c] == int(RoleFT.TARGET)
        for j in range(spec.q_tok):
            pos = base + spec.n_eff + j
            channel_idx[pos] = c; vslot_local[pos] = c
            span[pos] = qspan; role_ft[pos] = spec.role_ft[c]
            t_center[pos] = j * P_out + P_out * 0.5
            if is_target:
                # query [MASK] token — learned embedding, routed to the horizon head
                role[pos] = int(Role.QRY)
                content_state[pos] = int(ContentState.MASK)
                qry_pos.append(pos); qry_channel.append(c); qry_hidx.append(j)
            else:
                # known-future feature: observed token at a horizon timestamp,
                # encoded by the P_out-tier encoder like any observed token
                role[pos] = int(Role.CTX)
                vals, obs = _gather_window(z, rc, origin + j * P_out, P_out)
                content_state[pos] = int(ContentState.OBSERVED) if obs.any() else int(ContentState.NA)
                tier_pos[qspan].append(pos)
                tier_values[qspan].append(vals)
                tier_observed[qspan].append(obs)

    def _stack(lst, width):
        if lst:
            return np.stack(lst).astype(np.float32)
        return np.zeros((0, width), np.float32)

    return AssembledSegment(
        S=S, C=C,
        channel_idx=channel_idx, vslot_local=vslot_local, t_center=t_center,
        span=span, role=role, role_ft=role_ft, content_state=content_state, valid=valid,
        tier_pos=[np.asarray(p, np.int64) for p in tier_pos],
        tier_values=[_stack(tier_values[k], PATCH[k]) for k in range(V)],
        tier_observed=[_stack(tier_observed[k], PATCH[k]) for k in range(V)],
        qry_pos=np.asarray(qry_pos, np.int64),
        qry_channel=np.asarray(qry_channel, np.int64),
        qry_hidx=np.asarray(qry_hidx, np.int64),
        norm_a=norm_a, norm_sigma=norm_sigma, norm_sigma_tier=norm_sigma_tier,
        var_feats=var_feats,
    )
