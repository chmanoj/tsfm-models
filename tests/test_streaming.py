"""G4 streaming-from-disk corpus — shard format, zero-copy reader, rank-sharded
loader, varied synthetic generator, and a train smoke off the streamed shards.

The real GIFT-Eval pretrain converter is exercised opportunistically (skipped
when no pretrain tree is downloaded) — its only hard dep here is pyarrow (now a
core dep); gluonts seasonality degrades to -1 without the gifteval extra.
"""

import os

import numpy as np
import pytest
import torch

from tetris.config import load_config
from tetris.data.contract import build_loader, validate_item
from tetris.data.shards import ShardReader, ShardWriter, StreamingShardLoader
from tetris.data.synthetic_corpus import (
    DEFAULT_WEIGHTS, gen_kff_driver, gen_multi_seasonal, write_synthetic_corpus,
)
from tetris.train.shakedown import run_shakedown

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
PRETRAIN_ROOT = os.path.expanduser("~/Projects/gifteval/pretrain")


def _build(root, *, n=60, seed=3, shard_size=17, length_range=(64, 400)):
    with ShardWriter(root, shard_size=shard_size) as w:
        write_synthetic_corpus(w, n_series=n, seed=seed, length_range=length_range)
    return ShardReader(root)


# --- format / reader --------------------------------------------------------

def test_roundtrip_zero_copy_matches_known_array(tmp_path):
    d = str(tmp_path / "c")
    rng = np.random.default_rng(0)
    series = [rng.standard_normal((c, t)).astype(np.float32)
              for c, t in [(1, 50), (3, 120), (2, 33)]]
    with ShardWriter(d, shard_size=2) as w:           # force a shard boundary
        for k, s in enumerate(series):
            nf = 1 if s.shape[0] > 1 else 0
            w.add(s, nf, s.shape[0] - nf, source="t", kind="k", item_id=f"i{k}")
    r = ShardReader(d)
    assert r.n_series == 3
    assert len(r.manifest["shards"]) == 2             # 2 then 1
    for k, s in enumerate(series):
        back = r.read_array(k)
        assert back.shape == s.shape
        assert back.dtype == np.float32
        assert np.array_equal(back, s)                # exact bytes back


def test_read_yields_frozen_item_contract(tmp_path):
    r = _build(str(tmp_path / "c"))
    for g in range(r.n_series):
        item = r.read(g)
        validate_item(item)                           # 3-tuple, float32, rows==nf+nt
        data, nf, nt = item
        m = r.meta(g)
        assert data.shape == (m["C"], m["length"])
        assert nf == m["nf"] and nt == m["nt"]


def test_nan_allowed_inf_rejected(tmp_path):
    d = str(tmp_path / "c")
    with ShardWriter(d) as w:
        good = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        w.add(good, 0, 1)
        with pytest.raises(ValueError):
            w.add(np.array([[1.0, np.inf]], dtype=np.float32), 0, 1)
    back = ShardReader(d).read_array(0)
    assert np.isnan(back[0, 1]) and back[0, 0] == 1.0


# --- rank sharding (O6) -----------------------------------------------------

@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_rank_shards_disjoint_and_covering(tmp_path, world_size):
    d = str(tmp_path / "c")
    r = _build(d, n=50)
    N = r.n_series
    seen = []
    for rank in range(world_size):
        ld = StreamingShardLoader(d, rank=rank, world_size=world_size, cycle=False)
        idxs = list(range(rank, N, world_size))
        items = list(ld)
        assert len(items) == len(idxs) == len(ld)
        seen += idxs
    assert sorted(seen) == list(range(N))             # disjoint + covering


def test_loader_cycle_is_unbounded(tmp_path):
    d = str(tmp_path / "c")
    _build(d, n=8)
    ld = StreamingShardLoader(d, cycle=True)
    it = iter(ld)
    first = [next(it) for _ in range(20)]             # > n_series without StopIteration
    assert len(first) == 20


# --- determinism ------------------------------------------------------------

def test_corpus_is_deterministic(tmp_path):
    a = _build(str(tmp_path / "a"), n=40, seed=11)
    b = _build(str(tmp_path / "b"), n=40, seed=11)
    assert a.n_series == b.n_series
    for g in range(a.n_series):
        assert np.array_equal(a.read_array(g), b.read_array(g), equal_nan=True)
        assert a.meta(g)["kind"] == b.meta(g)["kind"]


# --- generator variety ------------------------------------------------------

def test_generator_families_and_metadata(tmp_path):
    from tetris.data.synthetic_corpus import _PRIMITIVE_SUBSET

    r = _build(str(tmp_path / "c"), n=400, seed=5)
    kinds = {r.meta(g)["kind"] for g in range(r.n_series)}
    # named families appear by name; the "univariate" family records the specific
    # primitive (e.g. random_walk) as its kind, so check those land too.
    named = set(DEFAULT_WEIGHTS) - {"univariate"}
    assert named <= kinds
    assert kinds & set(_PRIMITIVE_SUBSET)            # univariate primitives present
    # multivariate (shared_factor) and lead-lag covariate (kff_driver) carried
    has_mv = any(r.meta(g)["kind"] == "shared_factor" and r.meta(g)["C"] > 1
                 for g in range(r.n_series))
    has_kff = any(r.meta(g)["kind"] == "kff_driver" and r.meta(g)["nf"] >= 1
                  for g in range(r.n_series))
    assert has_mv and has_kff
    # seasonal families carry a positive integer season_length for MASE
    for g in range(r.n_series):
        m = r.meta(g)
        if m["kind"] in ("seasonal_known", "multi_seasonal"):
            assert m["season_length"] > 0


def test_multi_seasonal_and_kff_shapes():
    rng = np.random.default_rng(0)
    x, m = gen_multi_seasonal(rng, 1000)
    assert x.shape == (1000,) and m in (4, 7, 12, 24, 48, 96, 144, 168, 336, 52)
    data, nf, nt, season = gen_kff_driver(rng, 800)
    assert data.shape[0] == nf + nt and nt == 1 and nf >= 1 and season > 0


def test_synthetic_corpus_magnitude_is_fp32_safe():
    """Regression: a fixed per-step exp rate made gen_exp_trend explode at long
    lengths (~1e29 at n=4096), whose square overflows fp32 variance -> NaN/training
    divergence (seen on the G4 2k GPU run). The whole corpus must stay fp32-safe."""
    import numpy as np
    from tetris.data import synthetic_corpus as SC

    names, p = SC._family_picker(SC.DEFAULT_WEIGHTS)
    gmax = 0.0
    for idx in range(1500):
        rng = np.random.default_rng((0, 0x5917, idx))
        n = int(rng.integers(96, 4097))
        fam = names[int(rng.choice(len(names), p=p))]
        data, *_ = SC.gen_series(rng, n, fam)
        a = np.abs(np.asarray(data, np.float64))
        a = a[np.isfinite(a)]
        if a.size:
            gmax = max(gmax, float(a.max()))
    assert gmax < 1e7, f"corpus magnitude {gmax:.2e} risks fp32 variance overflow"


# --- train smoke off streamed shards ---------------------------------------

def test_train_smoke_off_streamed_shards(tmp_path):
    d = str(tmp_path / "corpus")
    _build(d, n=64, seed=1, length_range=(96, 512))
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.data.loader = "streaming"
    cfg.data.shard_root = d
    cfg.data.shard_cycle = True
    cfg.packing.L_pack = 256
    cfg.packing.buffers_per_step = 2
    torch.manual_seed(0)
    losses = run_shakedown(cfg, steps=6, device="cpu")
    assert len(losses) == 6
    assert all(np.isfinite(losses))


def test_build_loader_factory_key(tmp_path):
    d = str(tmp_path / "c")
    _build(d, n=10)
    cfg = load_config(os.path.join(CONFIG_DIR, "streaming_synth.yaml"))
    cfg.data.shard_root = d
    loader = build_loader(cfg, rank=0, world_size=2)
    assert isinstance(loader, StreamingShardLoader)
    assert len(loader) == len(range(0, 10, 2))


# --- pretrain converter channel decoding (download-free) --------------------

def _arrow_with(tmp_path, name, target_col, feat_col=None):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    cols = {"item_id": pa.array(["a", "b"]), "freq": pa.array(["H", "H"]),
            "target": target_col}
    if feat_col is not None:
        cols["past_feat_dynamic_real"] = feat_col
    tab = pa.table(cols)
    p = str(tmp_path / f"{name}.arrow")
    with pa.OSFile(p, "wb") as sink:
        with ipc.new_stream(sink, tab.schema) as w:   # HF uses the IPC *stream* format
            w.write_table(tab)
    return p


def test_pretrain_decodes_all_three_channel_encodings(tmp_path):
    """Regression: gluonts encodes channels as list<float> (univariate),
    fixed_size_list<list<float>> AND plain list<list<float>> (both multivariate).
    All three must decode; the plain list<list<float>> feature column previously
    crashed the converter."""
    import pyarrow as pa
    from tetris.data.gifteval_pretrain import iter_subdataset

    fseries = pa.list_(pa.float32())
    flist = pa.list_(fseries)
    ffsl = pa.list_(fseries, 2)

    # (1) univariate list<float> target, no features
    p1 = _arrow_with(tmp_path, "uni",
                     pa.array([[1.0, 2.0, 3.0], [4.0, 5.0]], type=fseries))
    rows1 = list(iter_subdataset(p1))
    assert len(rows1) == 2 and rows1[0][1:3] == (0, 1) and rows1[0][0].shape[0] == 1

    # (2) fixed_size_list<list<float>>[2] target + plain list<list<float>> features
    tgt2 = pa.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], type=ffsl)
    feat2 = pa.array([[[9.0, 9.0]], [[8.0, 8.0]]], type=flist)   # 1 feature channel, variable-list
    p2 = _arrow_with(tmp_path, "mv", tgt2, feat2)
    data, nf, nt, _m, _id = next(iter_subdataset(p2))
    assert (nf, nt) == (1, 2) and data.shape == (3, 2)          # features-first: 1 feat + 2 tgt


# --- real GIFT-Eval pretrain converter (opportunistic) ----------------------

@pytest.mark.skipif(not os.path.isdir(PRETRAIN_ROOT),
                    reason="no downloaded GiftEvalPretrain tree")
def test_pretrain_converter_roundtrip(tmp_path):
    from tetris.data.gifteval_pretrain import list_subdatasets, write_pretrain_corpus

    subs = list_subdatasets(PRETRAIN_ROOT)
    if not subs:
        pytest.skip("pretrain tree present but empty")
    d = str(tmp_path / "pre")
    with ShardWriter(d, shard_size=50) as w:
        n = write_pretrain_corpus(w, PRETRAIN_ROOT, subdatasets=subs[:3], max_series=40)
    assert n > 0
    r = ShardReader(d)
    assert r.n_series == n
    for g in range(r.n_series):
        item = r.read(g)
        validate_item(item)
        assert r.meta(g)["source"] == "pretrain"
