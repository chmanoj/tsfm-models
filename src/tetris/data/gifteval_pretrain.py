"""Convert downloaded GiftEvalPretrain sub-datasets into the G4 shard format.

GiftEvalPretrain ships one HuggingFace-``datasets`` Arrow tree per sub-dataset
(``<name>/data-*.arrow`` in gluonts layout). ``target`` is either ``list<float>``
(univariate) or ``fixed_size_list<list<float>>[C]`` (multivariate);
``past_feat_dynamic_real`` (when present) carries known/past covariates the same
way. We read each file with **pyarrow directly** (not HF ``datasets``, which
deserializes the whole list-column per row), stack channels **features-first** to
match the frozen ``Item`` contract, derive ``season_length`` from ``freq`` via
GIFT-Eval's own ``get_seasonality``, and hand each series to a :class:`ShardWriter`
— so synthetic and real pretrain land in **one** streaming corpus, read back by
the same zero-copy reader.

``pyarrow``/``gluonts`` are lazy here (only needed when materializing), matching
the rest of the GIFT-Eval seam — never imported in CI.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .gifteval_download import _season_length


def _read_table(path: str):
    """Read an HF/gluonts Arrow file (stream format, with a file-format fallback)
    fully into memory so its buffers outlive the file handle."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(path, "r") as src:
        try:
            tab = ipc.open_stream(src).read_all()
        except pa.lib.ArrowInvalid:
            src.seek(0)
            tab = ipc.open_file(src).read_all()
    # combine_chunks + a deep copy detaches from the mmap (the files are small).
    return tab.combine_chunks()


def _row_channels(col_chunked, i: int) -> List[np.ndarray]:
    """Return the per-channel float arrays of row ``i``. gluonts columns appear in
    three shapes: ``list<float>`` (1 channel, univariate), and two multivariate
    encodings — ``fixed_size_list<list<float>>[C]`` and plain ``list<list<float>>``
    (variable channel count). All resolve to ``scalar.values`` being either a float
    array (univariate) or a list-of-lists array (one sub-list per channel)."""
    import pyarrow as pa

    inner = col_chunked[i].values                   # Array for this row (zero-copy view)
    if pa.types.is_floating(inner.type) or pa.types.is_integer(inner.type):
        return [np.asarray(inner.to_numpy(zero_copy_only=False), dtype=np.float32)]
    # nested: each element is one channel (a list<float> sub-list)
    return [np.asarray(ch.values.to_numpy(zero_copy_only=False), dtype=np.float32)
            for ch in inner]


def iter_subdataset(path: str):
    """Yield ``(data[C, t] float32, nf, nt, season_length, item_id)`` for every
    series (row) of one sub-dataset Arrow file, features-first."""
    table = _read_table(path)
    names = set(table.schema.names)
    has_feat = "past_feat_dynamic_real" in names
    has_freq = "freq" in names
    tgt = table.column("target")
    feat = table.column("past_feat_dynamic_real") if has_feat else None
    fcol = table.column("freq") if has_freq else None
    icol = table.column("item_id") if "item_id" in names else None
    for i in range(table.num_rows):
        targets = _row_channels(tgt, i)
        features = _row_channels(feat, i) if has_feat else []
        chans = features + targets               # features-first
        t = min(c.shape[0] for c in chans)
        if t < 2:
            continue                             # degenerate series
        data = np.stack([c[:t] for c in chans]).astype(np.float32)
        nf, nt = len(features), len(targets)
        freq = fcol[i].as_py() if fcol is not None else None
        try:
            m = _season_length(freq) if freq else -1
        except Exception:
            m = -1
        item_id = icol[i].as_py() if icol is not None else f"row_{i}"
        yield data, nf, nt, int(m), str(item_id)


def list_subdatasets(root: str) -> List[str]:
    """Sub-dataset directory names under a downloaded GiftEvalPretrain ``root``."""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*.arrow")):
            out.append(d)
    return out


def write_pretrain_corpus(
    writer,
    root: str,
    *,
    subdatasets: Optional[List[str]] = None,
    source: str = "pretrain",
    max_series: Optional[int] = None,
) -> int:
    """Convert (a subset of) downloaded GiftEvalPretrain sub-datasets into shards
    via ``writer``. Returns the number of series written. ``max_series`` caps the
    total (for smokes); ``subdatasets`` restricts which directories are read."""
    root = os.path.expanduser(root)
    names = subdatasets if subdatasets is not None else list_subdatasets(root)
    written = 0
    for name in names:
        files = sorted(glob.glob(os.path.join(root, name, "*.arrow")))
        for f in files:
            for data, nf, nt, m, item_id in iter_subdataset(f):
                if max_series is not None and written >= max_series:
                    return written
                writer.add(data, nf, nt, season_length=m, source=source,
                           kind=name, item_id=f"{name}/{item_id}")
                written += 1
    return written
