"""Tests for the characterization helper (the learnability split). The panels need real
GIFT-Eval data so they are exercised manually; the MASE split is pure-numpy and tested here."""
import numpy as np

from tetris.data.synth_explore import mase_split


def test_mase_split_seasonal_naive_beats_last_on_clean_cycle():
    # a clean repeating daily cycle ⇒ seasonal-naive forecasts it, last-value does not.
    season = 24
    t = np.arange(24 * 30)
    x = np.sin(2 * np.pi * t / season)
    sn, last, lin = mase_split(x, season, season)
    assert np.isfinite(sn) and np.isfinite(last)
    assert sn < last                                     # the recurring pattern is learnable


def test_mase_split_nan_snaive_when_non_seasonal():
    # season < 2 ⇒ seasonal-naive is undefined (nan); last/linear are still computed.
    x = np.cumsum(np.ones(200)) + np.random.default_rng(0).normal(0, 0.01, 200)  # trend
    sn, last, lin = mase_split(x, 1, 12)
    assert np.isnan(sn) and np.isfinite(last) and np.isfinite(lin)
