"""Materialize a streaming corpus to disk (G4 writer entrypoint).

Generates the varied synthetic corpus once (seeded) and, optionally, converts a
downloaded GiftEvalPretrain subset into the **same** Arrow-IPC shard format, then
writes ``manifest.json`` + ``index.arrow`` + ``shard-*.arrow`` under ``--out``.
Run offline; the ``streaming`` loader then reads these shards with zero-copy
random access.

Examples::

    # 20k varied synthetic series -> ~8 shards
    python -m tetris.data.materialize --out outputs/corpus_synth --n-synthetic 20000

    # synthetic + real GIFT-Eval pretrain (already downloaded) in one corpus
    python -m tetris.data.materialize --out outputs/corpus_mixed \\
        --n-synthetic 20000 --pretrain-root ~/Projects/gifteval/pretrain
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from .shards import ShardReader, ShardWriter
from .synthetic_corpus import write_synthetic_corpus


def build_corpus(
    out: str,
    *,
    n_synthetic: int = 20000,
    seed: int = 0,
    shard_size: int = 2500,
    pretrain_root: Optional[str] = None,
    pretrain_subdatasets=None,
    pretrain_max_series: Optional[int] = None,
    length_range=(96, 4096),
) -> dict:
    """Write a synthetic (+ optional pretrain) corpus to ``out`` and return its
    manifest."""
    builder = dict(seed=seed, n_synthetic=n_synthetic, shard_size=shard_size,
                   length_range=list(length_range))
    n_pre = 0
    with ShardWriter(out, shard_size=shard_size, builder=builder) as w:
        write_synthetic_corpus(w, n_series=n_synthetic, seed=seed,
                               length_range=length_range)
        if pretrain_root:
            from .gifteval_pretrain import write_pretrain_corpus
            n_pre = write_pretrain_corpus(
                w, pretrain_root, subdatasets=pretrain_subdatasets,
                max_series=pretrain_max_series)
    builder["n_pretrain"] = n_pre
    return ShardReader(out).manifest


def main() -> None:  # pragma: no cover - manual entrypoint
    ap = argparse.ArgumentParser(description="Materialize a TETRIS streaming corpus")
    ap.add_argument("--out", required=True, help="output corpus directory")
    ap.add_argument("--n-synthetic", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=2500)
    ap.add_argument("--pretrain-root", default=None,
                    help="downloaded GiftEvalPretrain root (optional)")
    ap.add_argument("--pretrain-max-series", type=int, default=None)
    args = ap.parse_args()

    root = os.path.expanduser(args.pretrain_root) if args.pretrain_root else None
    manifest = build_corpus(
        args.out, n_synthetic=args.n_synthetic, seed=args.seed,
        shard_size=args.shard_size, pretrain_root=root,
        pretrain_max_series=args.pretrain_max_series)
    srcs = ", ".join(f"{s['name']}={s['n_series']}" for s in manifest["sources"])
    print(f"wrote {manifest['n_series']} series in {len(manifest['shards'])} shards "
          f"to {args.out}  [{srcs}]")


if __name__ == "__main__":  # pragma: no cover
    main()
