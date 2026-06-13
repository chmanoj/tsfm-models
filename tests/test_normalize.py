"""[GATE] test_normalize — D10 round-trip / inversion / fallback chain / per-tier σ.

Covers: invert(forward(x))≈x across synthetic series including extremes; the
sigma fallback chain (constant -> unit -> all-zeros z; intermittent -> mean|dx|);
per-tier sigma; loss-target/inverse consistency.
"""

import math

import torch

from tetris import normalize as N
from tetris.constants import V


def _series_bank(dtype=torch.float64):
    """Eight synthetic series types + extremes (D10 stress set)."""
    torch.manual_seed(0)
    n = 600
    t = torch.arange(n, dtype=dtype)
    bank = {}
    bank["linear_trend"] = 0.5 * t + torch.randn(n, dtype=dtype)
    bank["exp_trend"] = torch.exp(0.01 * t) + torch.randn(n, dtype=dtype)
    bank["seasonal"] = 10 * torch.sin(2 * math.pi * t / 24) + torch.randn(n, dtype=dtype)
    # AR(1)
    ar = torch.zeros(n, dtype=dtype)
    eps = torch.randn(n, dtype=dtype)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + eps[i]
    bank["ar1"] = ar
    # random walk
    bank["random_walk"] = torch.cumsum(torch.randn(n, dtype=dtype), dim=0)
    # regime jump
    rj = torch.randn(n, dtype=dtype)
    rj[n // 2:] += 1000.0
    bank["regime_jump"] = rj
    # extremes: very large scale
    bank["extreme_large"] = 1e8 * torch.randn(n, dtype=dtype) + 1e9
    # extremes: tiny scale
    bank["extreme_small"] = 1e-7 * torch.randn(n, dtype=dtype)
    return bank


def test_roundtrip_all_series():
    for name, x in _series_bank().items():
        st = N.compute_stats(x)
        z = N.forward(x, st.a, st.sigma)
        xr = N.invert(z, st.a, st.sigma)
        assert torch.isfinite(z).all(), name
        torch.testing.assert_close(xr, x, rtol=1e-9, atol=1e-6, msg=name)


def test_roundtrip_with_nan():
    x = _series_bank()["ar1"].clone()
    x[10:15] = float("nan")
    x[300] = float("nan")
    st = N.compute_stats(x)
    z = N.forward(x, st.a, st.sigma)
    xr = N.invert(z, st.a, st.sigma)
    obs = torch.isfinite(x)
    torch.testing.assert_close(xr[obs], x[obs], rtol=1e-9, atol=1e-6)
    # NaN positions stay NaN (missingness handled upstream by D7).
    assert torch.isnan(z[~obs]).all()


def test_constant_series_degrades_to_zeros():
    x = torch.full((200,), 7.5, dtype=torch.float64)
    st = N.compute_stats(x)
    assert st.fallback == "unit"
    assert float(st.sigma) == 1.0
    z = N.forward(x, st.a, st.sigma)
    torch.testing.assert_close(z, torch.zeros_like(z), atol=1e-12, rtol=0)
    # still exactly reversible
    torch.testing.assert_close(N.invert(z, st.a, st.sigma), x, atol=1e-9, rtol=0)


def test_intermittent_uses_mean_abs_fallback():
    # Croston-style: mostly zero, sparse nonzero -> median|dx| == 0, mean|dx| > 0.
    x = torch.zeros(300, dtype=torch.float64)
    x[::40] = torch.tensor([3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 7.0, 3.0], dtype=torch.float64)[: x[::40].numel()]
    diffs = x[1:] - x[:-1]
    assert float(torch.median(diffs.abs())) == 0.0  # median path degenerate
    st = N.compute_stats(x)
    assert st.fallback == "mean_abs"
    assert float(st.sigma) > 0.0


def test_per_tier_sigma_present_and_finite():
    x = _series_bank()["random_walk"]
    st = N.compute_stats(x)
    assert st.sigma_tier.shape == (V,)
    assert torch.isfinite(st.sigma_tier).all()
    assert (st.sigma_tier > 0).all()


def test_loss_target_inverse_consistency():
    x = _series_bank()["regime_jump"]
    st = N.compute_stats(x)
    # horizon target round-trips through the same anchor/scale
    y = x[-20:]
    zt = N.horizon_target(y, st.a, st.sigma)
    yr = N.horizon_invert(zt, st.a, st.sigma)
    torch.testing.assert_close(yr, y, rtol=1e-9, atol=1e-6)
    # aux target: arcsinh of local increment in tier vol units (tier 0)
    level = x[100]
    future = x[101:105]
    at = N.aux_target(future, level, st.sigma_tier[0])
    recon = torch.sinh(at) * st.sigma_tier[0] + level
    torch.testing.assert_close(recon, future, rtol=1e-9, atol=1e-6)


def test_anchor_is_recent_median():
    # Anchor must track the recent level, not the whole-history middle (D10).
    x = torch.cat([torch.zeros(500, dtype=torch.float64), torch.full((32,), 100.0, dtype=torch.float64)])
    st = N.compute_stats(x, anchor_window=32)
    assert abs(float(st.a) - 100.0) < 1e-9
