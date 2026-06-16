"""On-disk Arrow-IPC shard format + streaming reader/loader (G4).

Materializes a ragged time-series corpus to disk **once** (seeded, deterministic)
and streams it back with **fast random access**. Each series is stored as a single
flattened ``[C, t]`` row in an Arrow-IPC *file* (footer-indexed, memory-mapped);
one series is read by **zero-copy slicing the shard's contiguous ``float32`` values
buffer** — no HuggingFace-``datasets`` list deserialization, no other series
touched, nothing but the one series we want is pulled into memory.

A corpus directory holds:

* ``manifest.json`` — corpus-level: format version, per-source counts, the shard
  list, ``n_series``, and the builder args (for reproducibility).
* ``index.arrow`` — a columnar per-series index
  (``shard, row, C, nf, nt, length, season_length, source, kind, item_id``),
  memory-mapped; its row order **is** the global series order.
* ``shard-00000.arrow`` … — Arrow-IPC **file** format, schema
  ``{values: list<float32>}``, one row per series = data flattened row-major
  ``[C, t]``.

The format is **source-agnostic** (synthetic + GIFT-Eval pretrain write the same
shards), and the loader is **rank-sharded round-robin by global series index**
(O6, exactly like ``StandInPretrainLoader``), so the reservoir / DDP / checkpoint
re-shard seams keep working unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from .contract import Item, validate_item

# v2 (H1): added per-series ``crop_ctx``/``crop_p`` index columns for the
# noise-robustness fixed-window items (Path B). v1 corpora are still readable —
# the reader synthesizes the columns as -1 (= no fixed window) when absent.
FORMAT_VERSION = 2
_READABLE_VERSIONS = (1, 2)
# One row per series; all series in a shard share one contiguous float32 buffer.
SHARD_SCHEMA = pa.schema([("values", pa.list_(pa.float32()))])
# int32 list offsets cap a shard at ~2.1e9 floats; flush before that (and far
# below it by default) so offsets never overflow.
_MAX_FLOATS_PER_SHARD = 256_000_000

_INDEX_FIELDS = ("shard", "row", "C", "nf", "nt", "length", "season_length",
                 "source", "kind", "item_id", "crop_ctx", "crop_p")


class ShardWriter:
    """Accumulate ragged series and flush them to Arrow-IPC shards.

    ``add`` takes one series as a ``[C, t]`` ``float32`` array plus its
    ``num_features``/``num_targets`` split and optional metadata. A shard is
    flushed once it reaches ``shard_size`` series (or would overflow the int32
    offset budget). ``close`` writes ``index.arrow`` + ``manifest.json``. Global
    series order = insertion order, so rank-sharding and the reservoir cursor are
    deterministic regardless of how generation was parallelized upstream.
    """

    def __init__(self, root, *, shard_size: int = 2500,
                 max_floats_per_shard: int = _MAX_FLOATS_PER_SHARD,
                 builder: Optional[dict] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.max_floats = int(max_floats_per_shard)
        self.builder = dict(builder or {})
        self._buf: List[np.ndarray] = []   # flat float32 arrays pending in the open shard
        self._buf_floats = 0
        self._shards: List[Dict] = []      # [{path, n_series}]
        self._index: List[Dict] = []       # one dict per series (global order)
        self._sources: Dict[str, int] = {}

    def add(self, data: np.ndarray, num_features: int, num_targets: int, *,
            season_length: int = -1, source: str = "", kind: str = "",
            item_id: str = "", crop_ctx: int = -1, crop_p: int = -1) -> None:
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"series must be 2-D [C, t], got shape {tuple(data.shape)}")
        C, t = data.shape
        nf, nt = int(num_features), int(num_targets)
        if nf < 0 or nt < 1 or C != nf + nt:
            raise ValueError(f"bad split: C={C} != nf({nf})+nt({nt}) (need nt>=1, nf>=0)")
        # Allow NaN (missing values, D7) but reject +/-inf which would corrupt norm.
        if np.isinf(data).any():
            raise ValueError("series contains +/-inf; only NaN is allowed for missing values")
        flat = data.reshape(-1)
        if self._buf and (len(self._buf) >= self.shard_size
                          or self._buf_floats + flat.size > self.max_floats):
            self._flush()
        self._index.append(dict(
            shard=len(self._shards), row=len(self._buf), C=C, nf=nf, nt=nt,
            length=t, season_length=int(season_length),
            source=str(source), kind=str(kind), item_id=str(item_id),
            crop_ctx=int(crop_ctx), crop_p=int(crop_p),
        ))
        self._buf.append(flat)
        self._buf_floats += int(flat.size)
        self._sources[source] = self._sources.get(source, 0) + 1

    def _flush(self) -> None:
        if not self._buf:
            return
        arr = pa.array(self._buf, type=pa.list_(pa.float32()))
        name = f"shard-{len(self._shards):05d}.arrow"
        with pa.OSFile(str(self.root / name), "wb") as sink:
            with ipc.new_file(sink, SHARD_SCHEMA) as w:
                w.write(pa.record_batch([arr], schema=SHARD_SCHEMA))
        self._shards.append(dict(path=name, n_series=len(self._buf)))
        self._buf = []
        self._buf_floats = 0

    def close(self) -> dict:
        self._flush()
        cols = {k: [r[k] for r in self._index] for k in _INDEX_FIELDS}
        idx = pa.table({
            "shard": pa.array(cols["shard"], pa.int32()),
            "row": pa.array(cols["row"], pa.int32()),
            "C": pa.array(cols["C"], pa.int32()),
            "nf": pa.array(cols["nf"], pa.int32()),
            "nt": pa.array(cols["nt"], pa.int32()),
            "length": pa.array(cols["length"], pa.int32()),
            "season_length": pa.array(cols["season_length"], pa.int32()),
            "source": pa.array(cols["source"], pa.string()),
            "kind": pa.array(cols["kind"], pa.string()),
            "item_id": pa.array(cols["item_id"], pa.string()),
            "crop_ctx": pa.array(cols["crop_ctx"], pa.int32()),
            "crop_p": pa.array(cols["crop_p"], pa.int32()),
        })
        with pa.OSFile(str(self.root / "index.arrow"), "wb") as sink:
            with ipc.new_file(sink, idx.schema) as w:
                w.write_table(idx)
        manifest = dict(
            version=FORMAT_VERSION,
            n_series=len(self._index),
            shards=self._shards,
            sources=[{"name": k, "n_series": v} for k, v in self._sources.items()],
            builder=self.builder,
        )
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ShardReader:
    """Random-access reader over a shard corpus (zero-copy per-series slice).

    The index is memory-mapped once into compact numpy columns (no python list of
    items). Each shard is memory-mapped lazily on first touch and cached; reading
    series ``g`` slices that shard's single contiguous values buffer between the
    series' offsets and copies out **only those floats** — the rest of the shard
    (and every other shard) stays on disk / paged out.
    """

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("version") not in _READABLE_VERSIONS:
            raise ValueError(f"unsupported shard format version {self.manifest.get('version')!r}")
        self._index_mm = pa.memory_map(str(self.root / "index.arrow"), "r")
        itab = ipc.open_file(self._index_mm).read_all()
        self._shard = itab.column("shard").to_numpy()
        self._row = itab.column("row").to_numpy()
        self._C = itab.column("C").to_numpy()
        self._nf = itab.column("nf").to_numpy()
        self._nt = itab.column("nt").to_numpy()
        self._length = itab.column("length").to_numpy()
        self._season = itab.column("season_length").to_numpy()
        # v1 corpora predate the fixed-window columns → synthesize -1 (no fixed window).
        names = set(itab.schema.names)
        self._crop_ctx = (itab.column("crop_ctx").to_numpy() if "crop_ctx" in names
                          else np.full(itab.num_rows, -1, np.int32))
        self._crop_p = (itab.column("crop_p").to_numpy() if "crop_p" in names
                        else np.full(itab.num_rows, -1, np.int32))
        self._index_table = itab  # keep strings (source/kind/item_id) addressable
        self._n = itab.num_rows
        self._shard_paths = [s["path"] for s in self.manifest["shards"]]
        # shard_idx -> (mmap, child float32 Array, offsets np.int64) opened lazily.
        self._cache: Dict[int, tuple] = {}

    @property
    def n_series(self) -> int:
        return int(self._n)

    def _shard_data(self, si: int):
        ent = self._cache.get(si)
        if ent is None:
            mm = pa.memory_map(str(self.root / self._shard_paths[si]), "r")
            tab = ipc.open_file(mm).read_all()
            col = tab.column("values").combine_chunks()   # single-batch -> single chunk
            ent = (mm, col.values, col.offsets.to_numpy())
            self._cache[si] = ent
        return ent

    def read_array(self, gidx: int) -> np.ndarray:
        """Return one series as a fresh ``float32`` ``[C, t]`` numpy array."""
        si = int(self._shard[gidx])
        row = int(self._row[gidx])
        _mm, child, offs = self._shard_data(si)
        s, e = int(offs[row]), int(offs[row + 1])
        view = child.slice(s, e - s).to_numpy(zero_copy_only=True)  # mmap-backed view
        C = int(self._C[gidx])
        return np.array(view, dtype=np.float32).reshape(C, -1)      # copy out just this series

    def read(self, gidx: int) -> Item:
        """Return one series as the frozen training ``Item``."""
        data = self.read_array(gidx)
        item = (torch.from_numpy(data), int(self._nf[gidx]), int(self._nt[gidx]))
        validate_item(item)
        return item

    def crop_hint(self, gidx: int) -> Optional[tuple]:
        """``(ctx, horizon)`` fixed-window hint for the noise-robustness items, or
        ``None`` when this series uses ordinary random cropping (the default / all
        v1 series)."""
        ctx, p = int(self._crop_ctx[gidx]), int(self._crop_p[gidx])
        return (ctx, p) if ctx > 0 and p > 0 else None

    def meta(self, gidx: int) -> dict:
        return dict(
            shard=int(self._shard[gidx]), row=int(self._row[gidx]),
            C=int(self._C[gidx]), nf=int(self._nf[gidx]), nt=int(self._nt[gidx]),
            length=int(self._length[gidx]), season_length=int(self._season[gidx]),
            source=self._index_table.column("source")[gidx].as_py(),
            kind=self._index_table.column("kind")[gidx].as_py(),
            item_id=self._index_table.column("item_id")[gidx].as_py(),
            crop_ctx=int(self._crop_ctx[gidx]), crop_p=int(self._crop_p[gidx]),
        )


class StreamingShardLoader:
    """Rank-sharded streaming loader over a shard corpus (O6).

    Yields the frozen ``Item`` for global indices ``range(rank, N, world_size)`` —
    disjoint and covering across ranks, content keyed by global index only (so it
    is identical to ``StandInPretrainLoader``'s contract, just read from disk).
    ``cycle`` makes the stream unbounded (the run stops at ``steps``); the reservoir
    above handles shuffling, so the on-disk order is read deterministically.
    """

    def __init__(self, root, *, rank: int = 0, world_size: int = 1,
                 cycle: bool = True, seed: int = 0) -> None:
        self.reader = ShardReader(root)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.cycle = bool(cycle)
        self.seed = int(seed)

    @classmethod
    def from_cfg(cls, cfg, *, rank: int = 0, world_size: int = 1) -> "StreamingShardLoader":
        root = cfg.data.shard_root
        if not root:
            raise ValueError("data.shard_root must be set for the 'streaming' loader "
                             "(materialize a corpus with `python -m tetris.data.materialize`)")
        return cls(root, rank=rank, world_size=world_size,
                   cycle=cfg.data.shard_cycle, seed=cfg.run.seed)

    def __iter__(self):
        from .contract import HintedItem
        shard = range(self.rank, self.reader.n_series, self.world_size)
        while True:
            for idx in shard:
                item = self.reader.read(idx)
                hint = self.reader.crop_hint(idx)
                # Bake the per-item fixed window (H1 Path B) only when set; otherwise
                # yield the bare frozen Item so the random-crop path is unchanged.
                yield HintedItem(item, hint) if hint is not None else item
            if not self.cycle:
                return

    def __len__(self) -> int:
        return len(range(self.rank, self.reader.n_series, self.world_size))
