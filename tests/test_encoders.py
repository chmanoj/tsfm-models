"""S7 (sub-step 1) test_encoders — index-routed per-tier dispatch (D3/Stage 7)
and the D4 variate-ID basis.

Covers: OBSERVED tokens get exactly their tier-encoder output gathered from the
flat store (re-encode-and-compare); MASK/NA/pad slots get the learned vectors;
routing is collision-free and respects ENCODER_CAP; the variate basis is
orthonormal (n_var<=d) and the receipt+gather composes per token.
"""

import numpy as np
import torch

from tetris.constants import ContentState, PATCH
from tetris.model.encoders import TierEncoders
from tetris.model.variate_id import VariateID, sample_orthonormal_basis
from tetris.packing.collator import pack
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window

P_OUT = 16
L_PACK = 256
D = 32
BASE_PRIOR = (16, 16, 16, 16, 16, 8)


def _params(**kw):
    return SamplerParams(l_pack=L_PACK, p_out=P_OUT, tier_prior=BASE_PRIOR, **kw)


def _seg(C, T, seed, **kw):
    g = torch.Generator().manual_seed(seed)
    item = (torch.randn(C, T, generator=g).cumsum(dim=1).to(torch.float32), 0, C)
    rng = np.random.default_rng(seed)
    spec = sample_window(0, C, T, _params(**kw), rng)
    return assemble(item, spec, P_OUT)


def _batch(buffers):
    return pack(buffers, l_pack=L_PACK, p_out=P_OUT)


def test_observed_tokens_get_their_encoder_window():
    torch.manual_seed(0)
    b = _batch([[_seg(2, 300, 1), _seg(3, 220, 2)], [_seg(1, 1500, 3)]])
    enc = TierEncoders(d_model=D, encoder_cap=L_PACK)
    enc.eval()
    with torch.no_grad():
        content = enc(b)
    assert content.shape == (b.B, L_PACK, D)
    assert torch.isfinite(content).all()

    # For a sample of OBSERVED tokens, recompute the encoder output directly from
    # the store window at raw_start and compare to the scattered content slot.
    cs = b.content_state
    obs = (cs == int(ContentState.OBSERVED)).nonzero(as_tuple=False)
    checked = 0
    with torch.no_grad():
        for bi, li in obs[:: max(1, len(obs) // 40)]:
            k = int(b.tier_id[bi, li]); p_k = PATCH[k]
            rs = int(b.raw_start[bi, li])
            win = torch.stack(
                [b.norm_values[bi, rs:rs + p_k],
                 b.observed[bi, rs:rs + p_k].to(torch.float32)], dim=-1
            ).unsqueeze(0)                                   # [1, P_k, 2]
            ref = enc.encoders[k](win)[0]
            torch.testing.assert_close(content[bi, li], ref, rtol=1e-5, atol=1e-5)
            checked += 1
    assert checked > 0


def test_mask_na_pad_slots_get_learned_vectors():
    torch.manual_seed(0)
    # kff_reveal off, all-target horizon → MASK query slots; blank a window → NA.
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 1200, generator=g).cumsum(dim=1).to(torch.float32)
    x[0, 1050:1140] = float("nan")
    rng = np.random.default_rng(7)
    spec = sample_window(0, 1, 1200, _params(), rng)
    seg = assemble((x, 0, 1), spec, P_OUT)
    b = _batch([[seg]])
    enc = TierEncoders(d_model=D, encoder_cap=L_PACK)
    with torch.no_grad():
        content = enc(b)

    cs = b.content_state[0]
    is_mask = cs == int(ContentState.MASK)
    is_na = cs == int(ContentState.NA)
    assert is_mask.any()
    torch.testing.assert_close(
        content[0][is_mask], enc.mask_vec.expand(int(is_mask.sum()), D)
    )
    torch.testing.assert_close(
        content[0][is_na], enc.na_vec.expand(int(is_na.sum()), D)
    )
    # pad slots are content_state==NA → na_vec
    pad = b.sample_id[0] == -1
    assert pad.any()
    torch.testing.assert_close(content[0][pad], enc.na_vec.expand(int(pad.sum()), D))


def test_routing_respects_cap_and_is_unique():
    # With a small CAP some OBSERVED tokens are dropped (content stays 0, not
    # overwritten by MASK/NA); the kept count is min(M_k, CAP) and never doubles.
    torch.manual_seed(0)
    b = _batch([[_seg(4, 2000, 11)]])
    cap = 8
    enc = TierEncoders(d_model=D, encoder_cap=cap)
    with torch.no_grad():
        content = enc(b)
    cs = b.content_state[0]
    tier = b.tier_id[0]
    for k in range(len(PATCH)):
        routed = (cs == int(ContentState.OBSERVED)) & (tier == k)
        n_routed = int(routed.sum())
        # rows actually encoded = nonzero content among routed slots (<= cap)
        nonzero = (content[0][routed].abs().sum(-1) > 0).sum().item()
        assert nonzero <= cap
        assert nonzero == min(n_routed, cap)


def test_variate_basis_orthonormal_and_compose():
    B, n_var, d = 2, 16, 32
    gen = torch.Generator().manual_seed(3)
    basis = sample_orthonormal_basis(B, n_var, d, generator=gen)
    assert basis.shape == (B, n_var, d)
    for bi in range(B):
        gram = basis[bi] @ basis[bi].t()
        torch.testing.assert_close(gram, torch.eye(n_var), rtol=1e-4, atol=1e-4)

    # fallback (n_var > d): rows unit-norm, still finite
    big = sample_orthonormal_basis(1, d + 5, d, generator=gen)
    torch.testing.assert_close(big.norm(dim=-1), torch.ones(1, d + 5), rtol=1e-4, atol=1e-4)

    # compose: per-token id = basis[b, uid]; receipt added
    vid = VariateID(d_model=d)
    b = _batch([[_seg(3, 300, 21)]])
    nv = int(b.variate_uid.max()) + 1
    basis2 = sample_orthonormal_basis(b.B, nv, d, generator=gen)
    with torch.no_grad():
        out = vid(b.variate_uid, b.stats_a, b.stats_sigma, basis2)
    assert out.shape == (b.B, L_PACK, d)
    assert torch.isfinite(out).all()
