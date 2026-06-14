"""[REQUIRED] test_aux_boundary — raw-time aux validity + per-tier weighting (D6/D10).

Asserts the aux term is counted exactly where ``tier_id==k & valid_aux`` (and the
target steps are observed/in-range), zeroed when the raw-time target region crosses
the origin or runs off history, gathers the right window (``raw_start+P_k``), and
applies the per-tier ``aux_weights`` vector. Uses a hand-crafted Batch for exact
arithmetic, plus an assemble-based check that ``valid_aux`` itself is False at the
origin/history boundaries (the precise raw-time check, not the D9.5 span proxy).
"""

import numpy as np
import torch

from tetris.constants import ContentState, PATCH, Role
from tetris.losses import aux_loss_per_tier, compute_loss
from tetris.model.tetris import ModelOutput
from tetris.packing.collator import Batch
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window

P0 = PATCH[0]  # 4
ZEROW = [0.0] * 6


def _hand_batch(valid_aux_flags):
    """B=1, L=6, all tier-0 (P_k=4). norm_values = arange(R) so target windows are
    known exactly. Tokens 0..3 OBSERVED context (raw_start 0,4,8,12); 4 = query,
    5 = pad."""
    L, R = 6, 24
    z = lambda dt: torch.zeros(1, L, dtype=dt)
    nv = torch.arange(R, dtype=torch.float32).unsqueeze(0)        # [1, R]
    batch = Batch(
        sample_id=torch.tensor([[0, 0, 0, 0, 0, -1]], dtype=torch.int32),
        channel_idx=torch.zeros(1, L, dtype=torch.int32),
        t_center=z(torch.float32),
        tier_id=torch.zeros(1, L, dtype=torch.int8),             # all tier 0
        role=torch.tensor([[0, 0, 0, 0, int(Role.QRY), 0]], dtype=torch.int8),
        content_state=torch.tensor(
            [[0, 0, 0, 0, int(ContentState.MASK), int(ContentState.NA)]], dtype=torch.int8
        ),
        role_ft=z(torch.int8),
        raw_start=torch.tensor([[0, 4, 8, 12, -1, -1]], dtype=torch.int32),
        variate_uid=torch.zeros(1, L, dtype=torch.int32),
        valid_aux=torch.tensor([valid_aux_flags], dtype=torch.bool),
        norm_values=nv,
        observed=torch.ones(1, R, dtype=torch.bool),
        stats_a=z(torch.float32),
        stats_sigma=torch.ones(1, L, dtype=torch.float32),
        horizon_target=torch.zeros(1, L, P0, dtype=torch.float32),
        target_valid=torch.zeros(1, L, P0, dtype=torch.bool),
    )
    return batch


def _output(batch, aux0):
    """ModelOutput with horizon all-zero (no horizon contribution) and a given
    tier-0 aux prediction; other tiers zero."""
    L = batch.tier_id.shape[1]
    aux = [torch.zeros(1, L, PATCH[k]) for k in range(6)]
    aux[0] = aux0
    return ModelOutput(horizon=torch.zeros(1, L, batch.horizon_target.shape[-1]), aux=aux)


def test_gather_reads_next_patch_window():
    # token0 target = norm_values[0+4 : 0+8] = [4,5,6,7]; predict exactly -> 0 error.
    batch = _hand_batch([True, False, False, False, False, False])
    aux0 = torch.zeros(1, 6, P0)
    aux0[0, 0] = torch.tensor([4.0, 5.0, 6.0, 7.0])
    per_tier = aux_loss_per_tier(_output(batch, aux0).aux, batch)
    assert torch.isclose(per_tier[0], torch.tensor(0.0))

    # off-by-one prediction -> MAE exactly 1.0 over the 4 valid steps
    per_tier_off = aux_loss_per_tier(_output(batch, aux0 + 1.0).aux, batch)
    assert torch.isclose(per_tier_off[0], torch.tensor(1.0))


def test_invalid_tokens_excluded_and_boundary_flip():
    # Only token0 valid -> error from token1's wrong pred must NOT count.
    batch = _hand_batch([True, False, False, False, False, False])
    aux0 = torch.zeros(1, 6, P0)
    aux0[0, 0] = torch.tensor([4.0, 5.0, 6.0, 7.0])   # token0 exact
    aux0[0, 1] = torch.tensor([99.0, 99.0, 99.0, 99.0])  # token1 wildly wrong
    per_tier = aux_loss_per_tier(_output(batch, aux0).aux, batch)
    assert torch.isclose(per_tier[0], torch.tensor(0.0))   # token1 excluded (valid_aux False)

    # Flip token1 valid -> now its error contributes (boundary token included).
    batch2 = _hand_batch([True, True, False, False, False, False])
    per_tier2 = aux_loss_per_tier(_output(batch2, aux0).aux, batch2)
    assert per_tier2[0] > 0


def test_per_tier_weighting():
    batch = _hand_batch([True, True, False, False, False, False])
    aux0 = torch.zeros(1, 6, P0)
    aux0[0, 0] = torch.tensor([4.0, 5.0, 6.0, 7.0])
    aux0[0, 1] = torch.tensor([9.0, 10.0, 11.0, 12.0])  # token1 target = nv[8:12]
    out = _output(batch, aux0)

    base = compute_loss(out, batch, aux_weights=[1.0] + ZEROW[1:])
    half = compute_loss(out, batch, aux_weights=[0.5] + ZEROW[1:])
    zero = compute_loss(out, batch, aux_weights=ZEROW)
    # tier-0 unweighted MAE is independent of the weight; weighted total scales.
    torch.testing.assert_close(base.aux_per_tier[0], half.aux_per_tier[0])
    torch.testing.assert_close(half.aux_total, 0.5 * base.aux_total)
    torch.testing.assert_close(zero.aux_total, torch.tensor(0.0))
    # horizon is all-zero here, so total == aux_total
    torch.testing.assert_close(base.total, base.aux_total)


def test_unobserved_target_steps_dropped():
    # Mark token0's target steps unobserved -> excluded even though valid_aux True.
    batch = _hand_batch([True, False, False, False, False, False])
    batch.observed[0, 4:8] = False
    aux0 = torch.zeros(1, 6, P0)
    aux0[0, 0] = torch.tensor([99.0, 99.0, 99.0, 99.0])  # wrong, but steps unobserved
    per_tier = aux_loss_per_tier(_output(batch, aux0).aux, batch)
    assert torch.isclose(per_tier[0], torch.tensor(0.0))


def test_assemble_valid_aux_false_at_origin_boundary():
    # The most-recent context token of a tier whose target [re, re+P_k) crosses the
    # origin must have valid_aux False (origin-crossing is the horizon's job).
    params = SamplerParams(l_pack=256, p_out=16, tier_prior=(16, 16, 16, 16, 16, 8))
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 3000, generator=g).cumsum(dim=1).to(torch.float32)
    rng = np.random.default_rng(4)
    spec = sample_window(0, 1, 3000, params, rng)
    seg = assemble((x, 0, 1), spec, 16)
    import tetris.telescope as TS
    cov = TS.coverage(spec.counts)
    store_lo = spec.origin - cov
    found_boundary = False
    for pos in range(seg.S):
        if seg.content_state[pos] != int(ContentState.OBSERVED) or seg.t_center[pos] >= 0:
            continue
        k = int(seg.tier_id[pos]); p_k = PATCH[k]
        rs_seg = int(seg.raw_start[pos])              # segment-store offset
        re_raw = store_lo + rs_seg + p_k              # raw end of this token's window
        tgt_hi = re_raw + p_k                          # raw end of the aux target
        if tgt_hi > spec.origin:                       # target crosses the origin
            assert not bool(seg.valid_aux[pos])
            found_boundary = True
    assert found_boundary
