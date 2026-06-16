"""H1 synth-v2 + quality-harness tests (all offline — no gluonts / real data).

Covers: the canonical feature battery; the fixed-window seam (sampler determinism +
unchanged random path, HintedItem → reservoir → clean horizon GT, shard v2 round-trip
+ v1 read-back compat); the new generators (KernelSynth-MV, SDEs, noise-robust,
dilution) shapes/determinism/tags; TestProfile fit/serialize + targeted-vs-general
overlap; and the Tier-1 harness metrics (C2ST/KS/MMD/AUC/verdict).
"""

import json
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from tetris.data import features as F
from tetris.data import quality_harness as Q
from tetris.data import synthetic_v2 as V
from tetris.data import synthetic_targeted as TGT
from tetris.data.contract import HintedItem, as_item
from tetris.data.shards import ShardReader, ShardWriter, StreamingShardLoader
from tetris.data.test_profile import FitRecord, TestProfile
from tetris.tokenize.window_sampler import FixedWindow, SamplerParams, sample_window
from tetris.packing.reservoir import StreamingReservoir


# --- features ---------------------------------------------------------------

def test_feature_vector_finite_and_deterministic():
    rng = np.random.default_rng(0)
    for x in [np.sin(np.arange(500) / 4), rng.normal(0, 1, 500), np.full(300, 7.0),
              np.array([1.0, 2.0]), np.full(50, np.nan)]:
        f = F.series_features(x)
        assert f.shape == (F.N_FEATURES,)
        assert np.isfinite(f).all()
    a = F.series_features(np.sin(np.arange(400) / 3))
    b = F.series_features(np.sin(np.arange(400) / 3))
    assert np.array_equal(a, b)


def test_features_discriminate_known_signals():
    n = 512
    sine = F.series_features(np.sin(2 * np.pi * np.arange(n) / 24))
    noise = F.series_features(np.random.default_rng(1).normal(0, 1, n))
    se = F.FEATURE_NAMES.index("seasonal_strength")
    spe = F.FEATURE_NAMES.index("spectral_entropy")
    assert sine[se] > 0.5 > noise[se]            # sine is seasonal, noise is not
    assert noise[spe] > sine[spe]                # white noise has higher spectral entropy


# --- fixed-window seam ------------------------------------------------------

def test_sample_window_fixed_is_deterministic_crop():
    params = SamplerParams(l_pack=512, p_out=16)
    rng = np.random.default_rng(0)
    spec = sample_window(0, 1, 200, params, rng, fixed=FixedWindow(120, 32))
    assert spec.origin == 120
    assert spec.p == min(32, 4 * 16)             # clamped to the per-pass token budget


def test_sample_window_default_path_unchanged():
    # Without `fixed`, two draws from the same rng state are identical (the random
    # path is untouched by the additive parameter).
    params = SamplerParams(l_pack=512, p_out=16)
    a = sample_window(0, 1, 300, params, np.random.default_rng(5))
    b = sample_window(0, 1, 300, params, np.random.default_rng(5))
    assert (a.origin, a.p) == (b.origin, b.p)


def test_fixed_window_yields_clean_horizon_through_reservoir():
    # noisy context + constant (clean) horizon; the reservoir must crop at the baked
    # boundary so the horizon GT region is the clean constant.
    d = tempfile.mkdtemp()
    ctx, hor = 96, 32
    rng = np.random.default_rng(3)
    series = np.concatenate([np.cumsum(rng.normal(0, 1, ctx)),
                             np.full(hor, 4.2)])[None, :].astype(np.float32)
    with ShardWriter(d) as w:
        for _ in range(6):
            w.add(series, 0, 1, source="synth_general", kind="nr_test",
                  crop_ctx=ctx, crop_p=hor)
    res = StreamingReservoir(StreamingShardLoader(d, cycle=False),
                             SamplerParams(l_pack=512, p_out=16),
                             l_pack=512, p_out=16, buffers_per_step=2, reservoir_k=6)
    res._topup_reservoir()
    item, spec = res._reservoir[0]
    assert spec.origin == ctx
    horizon_vals = item[0].numpy()[0][spec.origin:spec.origin + spec.p]
    assert np.allclose(horizon_vals, 4.2)


def test_shard_v2_roundtrip_and_crop_hint():
    d = tempfile.mkdtemp()
    with ShardWriter(d) as w:
        w.add(np.ones((1, 50), np.float32), 0, 1, crop_ctx=30, crop_p=10)
        w.add(np.ones((1, 50), np.float32), 0, 1)               # no hint
    r = ShardReader(d)
    assert r.manifest["version"] == 2
    assert r.crop_hint(0) == (30, 10)
    assert r.crop_hint(1) is None
    assert r.meta(0)["crop_ctx"] == 30


def test_shard_v1_read_back_compat():
    # Forge a v1 corpus (index without crop columns, manifest version=1) and confirm
    # the reader loads it with crop hints synthesized as None.
    d = tempfile.mkdtemp()
    with ShardWriter(d) as w:
        w.add(np.arange(20, dtype=np.float32)[None, :], 0, 1, item_id="a")
    import pathlib
    root = pathlib.Path(d)
    man = json.loads((root / "manifest.json").read_text())
    man["version"] = 1
    (root / "manifest.json").write_text(json.dumps(man))
    tab = ipc.open_file(pa.memory_map(str(root / "index.arrow"), "r")).read_all()
    keep = [n for n in tab.schema.names if n not in ("crop_ctx", "crop_p")]
    tab = tab.select(keep)
    with pa.OSFile(str(root / "index.arrow"), "wb") as sink:
        with ipc.new_file(sink, tab.schema) as wr:
            wr.write_table(tab)
    r = ShardReader(d)
    assert r.manifest["version"] == 1
    assert r.crop_hint(0) is None
    assert r.read_array(0).shape == (1, 20)


def test_hinted_item_helpers():
    item = (torch.zeros(1, 10), 0, 1)
    assert as_item(item) is item
    h = HintedItem(item, (5, 3))
    assert as_item(h) is item and h.fixed_window == (5, 3)


# --- generators -------------------------------------------------------------

def test_kernelsynth_multivariate_shape_and_correlation():
    x = V.gen_kernelsynth(np.random.default_rng(0), 600, 6)
    assert x.shape == (6, 600) and np.isfinite(x).all()
    cm = np.corrcoef(x)
    off = cm[~np.eye(6, dtype=bool)]
    assert np.abs(off).max() > 0.3                # channels correlate through factors


def test_sde_and_noise_robust_finite():
    rng = np.random.default_rng(1)
    for fn in (V.gen_ou, V.gen_jump_diffusion, V.gen_vol_cluster):
        assert np.isfinite(fn(rng, 400)).all()
    for k in V._NOISE_ROBUST_KINDS:
        s, kind = V.gen_noise_robust(np.random.default_rng(2), 200, 32, kind=k)
        assert s.shape == (232,) and np.isfinite(s).all()
        # the clean horizon is far smoother than the noisy context
        ctx_rough = np.abs(np.diff(s[:200], 2)).mean()
        hor_rough = np.abs(np.diff(s[200:], 2)).mean()
        assert hor_rough < ctx_rough


def test_dilution_adds_features_keeps_targets():
    data = np.random.default_rng(3).normal(0, 1, (2, 100)).astype(np.float32)
    aug, nf, nt = V.inject_dilution(np.random.default_rng(3), data, 0, 2,
                                    prob=1.0, max_extra=3)
    assert nt == 2 and nf >= 1 and aug.shape[0] == nf + nt
    assert np.array_equal(aug[nf:], data)         # targets are the trailing rows, intact


def test_general_corpus_tags_and_determinism():
    def build():
        d = tempfile.mkdtemp()
        with ShardWriter(d, shard_size=500) as w:
            V.write_general_corpus(w, n_series=200, seed=0)
        return ShardReader(d)
    r1, r2 = build(), build()
    assert all(np.array_equal(r1.read_array(g), r2.read_array(g), equal_nan=True)
               for g in range(200))
    sources = {r1.meta(g)["source"] for g in range(200)}
    assert sources == {"synth_general"}
    # at least some noise-robust items carry a fixed-window crop hint
    assert any(r1.crop_hint(g) is not None for g in range(200))


# --- TestProfile + targeted -------------------------------------------------

def _stub_profile(seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(40):
        n = int(rng.integers(300, 700))
        x = 3 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 0.7, n)
        recs.append(FitRecord(feats=F.channel_features(x[None, :]), freq="H",
                              season=24, horizon=48, n_channels=1))
    return TestProfile.fit(recs)


def test_profile_fit_save_load_roundtrip():
    prof = _stub_profile()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        prof.save(fh.name)
        re = TestProfile.load(fh.name)
    assert set(re.groups) == set(prof.groups)
    assert re.groups["H"]["weight"] == 40


def test_targeted_matches_profile_better_than_general():
    prof = _stub_profile()
    real = [3 * np.sin(2 * np.pi * np.arange(500) / 24)
            + np.random.default_rng(k).normal(0, 0.7, 500) for k in range(40)]
    tgt = [TGT.gen_targeted(np.random.default_rng((9, i)), prof)[0][0] for i in range(40)]
    gen = [V.gen_general_series(np.random.default_rng((5, i)), 500, "kernelsynth")[0][0]
           for i in range(40)]
    rf, tf, gf = (F.feature_matrix(real), F.feature_matrix(tgt), F.feature_matrix(gen))
    se = F.FEATURE_NAMES.index("seasonal_strength")
    # targeted recovers the seasonal structure; generic kernelsynth does not
    assert abs(tf[:, se].mean() - rf[:, se].mean()) < abs(gf[:, se].mean() - rf[:, se].mean())


def test_targeted_corpus_writes_tagged_series():
    prof = _stub_profile()
    d = tempfile.mkdtemp()
    with ShardWriter(d, shard_size=200) as w:
        TGT.write_targeted_corpus(w, n_series=60, profile=prof, seed=0)
    r = ShardReader(d)
    assert r.n_series == 60
    assert {r.meta(g)["source"] for g in range(60)} == {"synth_targeted"}
    assert all(np.isfinite(a[np.isfinite(a)]).all()
               for a in (r.read_array(g) for g in range(60)))


# --- Tier-1 harness metrics -------------------------------------------------

def test_auc_and_c2st_extremes():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(100), np.ones(100)]
    assert abs(Q.auc_score(rng.normal(0, 1, 200), labels) - 0.5) < 0.15
    sep = np.r_[rng.normal(-3, 0.4, 100), rng.normal(3, 0.4, 100)]
    assert Q.auc_score(sep, labels) > 0.95
    A, B = rng.normal(0, 1, (300, 6)), rng.normal(0, 1, (300, 6))
    C = rng.normal(6, 1, (300, 6))
    assert abs(Q.c2st_auc(A, B, seed=1) - 0.5) < 0.15
    assert Q.c2st_auc(A, C, seed=1) > 0.9


def test_ks_mmd_and_evaluate_verdict():
    rng = np.random.default_rng(0)
    real = rng.normal(0, 1, (300, F.N_FEATURES))
    close = rng.normal(0, 1, (300, F.N_FEATURES))
    far = rng.normal(5, 1, (300, F.N_FEATURES))
    assert Q.ks_2samp(real[:, 0], far[:, 0]) > Q.ks_2samp(real[:, 0], close[:, 0])
    assert Q.rbf_mmd(real, far) > Q.rbf_mmd(real, close)
    res = Q.evaluate(real, {"targeted": close, "general": far, "noise": far + 10}, seed=0)
    assert res.families["targeted"]["c2st_dynamics"] < res.families["general"]["c2st_dynamics"]
    assert "c2st_knn_dynamics" in res.families["targeted"]
    assert "dyn=" in res.verdict


def test_knn_c2st_and_dynamics_subset():
    rng = np.random.default_rng(0)
    A, B = rng.normal(0, 1, (200, F.N_FEATURES)), rng.normal(0, 1, (200, F.N_FEATURES))
    C = rng.normal(6, 1, (200, F.N_FEATURES))
    assert abs(Q.knn_c2st_auc(A, B, seed=1) - 0.5) < 0.2
    assert Q.knn_c2st_auc(A, C, seed=1) > 0.9
    # dynamics subset excludes the superficial length/scale axes
    assert "log_length" not in F.DYNAMICS_FEATURES and "log_scale" not in F.DYNAMICS_FEATURES
    assert len(F.DYNAMICS_IDX) == F.N_FEATURES - 2
