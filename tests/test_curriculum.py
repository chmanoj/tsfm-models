"""G5 curriculum loader — progress-annealed weighted source mixture (D13).

CI-safe: component loaders are tiny synthetic streaming corpora + an in-process
fake "train" loader, so the mixing schedule, temperature/size weighting, phase-1→
phase-2 anneal, determinism, rank decorrelation, and the ``build_loader`` factory
key are all exercised offline.
"""

from __future__ import annotations

import os
from collections import Counter
from itertools import islice

import numpy as np
import pytest
import torch

from tetris.config import CurriculumCfg, SourceCfg, load_config
from tetris.data.contract import build_loader, validate_item
from tetris.data.curriculum import CurriculumLoader, _Source
from tetris.data.shards import ShardWriter
from tetris.data.synthetic_corpus import write_synthetic_corpus

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


class _Tagged:
    """Infinite loader yielding 1-channel Items tagged by a source id (the constant
    fill value), so a drawn item's source is recoverable from its data."""

    def __init__(self, tag: int) -> None:
        self.tag = tag

    def __iter__(self):
        while True:
            yield (torch.full((1, 8), float(self.tag), dtype=torch.float32), 0, 1)


def _src(name, tag, *, size, m1, m2):
    return _Source(name=name, loader=_Tagged(tag), size=size, mult1=m1, mult2=m2)


def _curric(components, **kw):
    kw.setdefault("total_items", 1000)
    kw.setdefault("phase2_start", 0.8)
    kw.setdefault("alpha", 0.0)            # size-neutral unless a test sets it
    kw.setdefault("temperature", 1.0)
    return CurriculumLoader(components, **kw)


# --- schedule / weights -----------------------------------------------------

def test_phase1_weights_are_size_alpha_times_multiplier():
    # alpha=0.5: weight ∝ multiplier × sqrt(size)
    c = _curric([_src("a", 1, size=100, m1=1.0, m2=1.0),
                 _src("b", 2, size=4, m1=2.0, m2=2.0)], alpha=0.5)
    w = c.weights_at(0.0)
    raw = np.array([1.0 * 10.0, 2.0 * 2.0])     # sqrt(100)=10, sqrt(4)=2
    assert np.allclose(w, raw / raw.sum())


def test_phase2_anneal_interpolates_multiplier():
    c = _curric([_src("broad", 1, size=1, m1=1.0, m2=0.2),
                 _src("train", 2, size=1, m1=0.1, m2=1.0)],
                phase2_start=0.5, alpha=0.0)
    w0 = c.weights_at(0.0)                       # phase 1
    assert np.allclose(w0, np.array([1.0, 0.1]) / 1.1)
    w_mid = c.weights_at(0.75)                   # halfway through the anneal
    mult = np.array([1.0 + 0.5 * (0.2 - 1.0), 0.1 + 0.5 * (1.0 - 0.1)])  # [0.6, 0.55]
    assert np.allclose(w_mid, mult / mult.sum())
    w1 = c.weights_at(1.0)                       # fully annealed -> train upweighted
    assert np.allclose(w1, np.array([0.2, 1.0]) / 1.2)
    assert w1[1] > w0[1]                         # train share grew


def test_zero_multiplier_removes_source_and_renormalizes():
    c = _curric([_src("a", 1, size=1, m1=0.0, m2=1.0),
                 _src("b", 2, size=1, m1=1.0, m2=1.0)], alpha=0.0)
    w = c.weights_at(0.0)
    assert w[0] == 0.0 and np.isclose(w[1], 1.0)


def test_temperature_flattens_mixture():
    comps = [_src("a", 1, size=1, m1=9.0, m2=9.0), _src("b", 2, size=1, m1=1.0, m2=1.0)]
    sharp = _curric(comps, temperature=1.0).weights_at(0.0)
    flat = _curric(comps, temperature=4.0).weights_at(0.0)
    # higher temperature pulls the dominant source down toward uniform
    assert flat[0] < sharp[0] and flat[1] > sharp[1]


# --- sampling behavior ------------------------------------------------------

def test_empirical_mix_matches_weights_and_shifts_across_phases():
    c = _curric([_src("broad", 1, size=1, m1=1.0, m2=0.0),
                 _src("train", 2, size=1, m1=0.0, m2=1.0)],
                total_items=4000, phase2_start=0.5, alpha=0.0)
    items = list(islice(iter(c), 4000))
    tags = [int(it[0][0, 0].item()) for it in items]
    early = Counter(tags[:1000])                # deep phase 1 -> all broad
    late = Counter(tags[-100:])                 # end of anneal -> broad mult →0
    assert early[1] == 1000 and early[2] == 0   # phase 1 is broad-only
    assert late[2] >= 90 and late[2] > late[1]  # tail is dominated by the train split
    for it in items:
        validate_item(it)


def test_deterministic_for_fixed_seed():
    mk = lambda: _curric([_src("a", 1, size=1, m1=1.0, m2=1.0),
                          _src("b", 2, size=1, m1=1.0, m2=1.0)], seed=7)
    a = [int(it[0][0, 0]) for it in islice(iter(mk()), 200)]
    b = [int(it[0][0, 0]) for it in islice(iter(mk()), 200)]
    assert a == b


def test_rank_rng_decorrelates_draws():
    mk = lambda r: _curric([_src("a", 1, size=1, m1=1.0, m2=1.0),
                            _src("b", 2, size=1, m1=1.0, m2=1.0)],
                           seed=0, rank=r, world_size=2)
    a = [int(it[0][0, 0]) for it in islice(iter(mk(0)), 200)]
    b = [int(it[0][0, 0]) for it in islice(iter(mk(1)), 200)]
    assert a != b                               # different source-draw streams


# --- full wiring via build_loader -------------------------------------------

def _write_corpus(root, *, n, seed):
    with ShardWriter(root, shard_size=8) as w:
        write_synthetic_corpus(w, n_series=n, seed=seed, length_range=(64, 256))


def test_build_loader_curriculum_mixes_two_streaming_corpora(tmp_path):
    synth = str(tmp_path / "synth")
    pre = str(tmp_path / "pre")
    _write_corpus(synth, n=20, seed=1)
    _write_corpus(pre, n=12, seed=2)
    cfg = load_config(os.path.join(CONFIG_DIR, "streaming_synth.yaml"))
    cfg.data.loader = "curriculum"
    cfg.curriculum = CurriculumCfg(
        sources=[
            SourceCfg(name="synthetic", loader="streaming", shard_root=synth,
                      multiplier_phase1=1.0, multiplier_phase2=0.3),
            SourceCfg(name="pretrain", loader="streaming", shard_root=pre,
                      multiplier_phase1=1.0, multiplier_phase2=1.0),
        ],
        total_items=500, phase2_start=0.8, alpha=0.3, temperature=1.0,
    )
    loader = build_loader(cfg)
    assert isinstance(loader, CurriculumLoader)
    # size term read from each corpus' manifest n_series (20 vs 12)
    sizes = {s.name: s.size for s in loader.components}
    assert sizes == {"synthetic": 20, "pretrain": 12}
    got = list(islice(iter(loader), 50))
    assert len(got) == 50
    for it in got:
        validate_item(it)


def test_curriculum_config_validation_rejects_empty_sources():
    cfg = load_config(os.path.join(CONFIG_DIR, "streaming_synth.yaml"))
    with pytest.raises(ValueError, match="curriculum.sources"):
        cfg.data.loader = "curriculum"
        cfg.__post_init__()


def test_train_smoke_reservoir_path_with_crop_schedule(tmp_path):
    """End-to-end reservoir path: curriculum mix of two corpora + the phase-2
    crop schedule (manual horizons -> no network), a few CPU train steps."""
    from tetris.train.loop import run_training

    synth = str(tmp_path / "synth")
    pre = str(tmp_path / "pre")
    _write_corpus(synth, n=24, seed=1)
    _write_corpus(pre, n=16, seed=2)
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.data.loader = "curriculum"
    cfg.packing.L_pack = 256
    cfg.packing.buffers_per_step = 2
    cfg.packing.reservoir_k = 32          # K-fill crosses phase2_start within the smoke
    cfg.packing.scheduler_window = 4
    cfg.curriculum = CurriculumCfg(
        sources=[
            SourceCfg(name="synthetic", loader="streaming", shard_root=synth,
                      multiplier_phase1=1.0, multiplier_phase2=0.2),
            SourceCfg(name="pretrain", loader="streaming", shard_root=pre,
                      multiplier_phase1=1.0, multiplier_phase2=1.0),
        ],
        total_items=40, phase2_start=0.3,
        phase2_crop_distribution="auto_from_test_configs",
        phase2_crop_horizons=[16, 32])    # manual -> offline
    cfg.__post_init__()
    torch.manual_seed(0)
    losses = run_training(cfg, steps=6, device="cpu")
    assert len(losses) == 6
    assert all(np.isfinite(losses))
