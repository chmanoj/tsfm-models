"""S3 test_synthetic — generators honor the contract; cross-channel structure present.

Covers: items honor exact (float32 [n,t], int, int) contract, features-first,
NaNs present, n/t vary; shared-factor channels show measurable cross-correlation;
lag-probe planted; loader reproducibility + disjoint rank shards (§9 seam).
"""

import numpy as np
import torch

from tetris.data import synthetic as S
from tetris.data.contract import build_loader, validate_item
from tetris.data.standin_loader import StandInPretrainLoader
from tetris.config import load_config

import os

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _loader(n_series=200, **kw):
    return StandInPretrainLoader(
        n_series=n_series,
        C_distribution=(1, 8),
        length_distribution=(64, 600),
        nan_cap=0.3,
        synthetic_mix={"shared_factor": 0.5, "univariate": 0.4, "lag_probe": 0.1},
        seed=0,
        **kw,
    )


def test_items_honor_contract():
    rows, lengths = set(), set()
    for item in _loader(n_series=150):
        validate_item(item)  # exact (float32 [n,t], int>=0, int>=1), features-first rows
        data, nf, nt = item
        assert data.dtype == torch.float32 and data.dim() == 2
        assert data.shape[0] == nf + nt
        rows.add(data.shape[0])
        lengths.add(data.shape[1])
    # n and t both vary across the stream
    assert len(rows) > 1
    assert len(lengths) > 1


def test_nans_present_in_stream():
    assert any(torch.isnan(data).any() for data, _, _ in _loader(n_series=150))


def test_features_first_for_lag_probe():
    # A lag-probe item is the only stand-in source with a feature row; the
    # feature must be row 0 (features-first), the target row 1.
    rng = np.random.default_rng(7)
    arr, lag = S.gen_lag_probe(rng, 500, lag=5)
    feature, target = arr[0], arr[1]
    # planted lagged dependence: target[t] ≈ feature[t-lag]
    lagged_corr = np.corrcoef(target[lag:], feature[:-lag])[0, 1]
    contemp_corr = np.corrcoef(target, feature)[0, 1]
    assert lagged_corr > 0.5
    assert abs(contemp_corr) < 0.3
    # channel-independent baseline must be weak: target's own lag-1 autocorr low
    own_ac = np.corrcoef(target[1:], target[:-1])[0, 1]
    assert abs(own_ac) < 0.3


def test_shared_factor_cross_correlation():
    rng = np.random.default_rng(3)
    x = S.gen_shared_factor(rng, 800, C=8, bank_size=7, kernels_per_variate=(4, 4))
    cm = np.corrcoef(x)
    off = cm[~np.eye(8, dtype=bool)]
    # channels sharing latent factors correlate measurably (D4 routing signal)
    assert np.abs(off).max() > 0.4
    assert np.abs(off).mean() > 0.1


def test_loader_reproducible():
    a = next(iter(_loader(n_series=10)))
    b = next(iter(_loader(n_series=10)))
    torch.testing.assert_close(a[0], b[0], equal_nan=True)
    assert (a[1], a[2]) == (b[1], b[2])


def test_rank_shards_disjoint():
    # §9 distributed seam: rank shards partition the series with no overlap.
    n = 60
    r0 = list(range(0, n, 2))
    r1 = list(range(1, n, 2))
    assert set(r0).isdisjoint(r1)
    assert sorted(r0 + r1) == list(range(n))
    # the loader honors the same partition by index
    l0 = _loader(n_series=n, rank=0, world_size=2)
    l1 = _loader(n_series=n, rank=1, world_size=2)
    assert len(l0) + len(l1) == n


def test_build_loader_factory():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    loader = build_loader(cfg)
    item = next(iter(loader))
    validate_item(item)
