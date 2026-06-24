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

    # H1.1 TARGETED corpus: N series per GIFT-Eval test config from its validated recipe
    # (data resembling the test split; each series tagged kind=<config>).
    python -m tetris.data.materialize --out artifacts/corpus_recipe --n-recipe 50
    # then validate per config (needs $GIFT_EVAL):
    #   python -m tetris.data.synth_explore validate-corpus artifacts/corpus_recipe \\
    #       --out-dir artifacts/corpus_validation
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


def build_corpus_v2(
    out: str,
    *,
    n_general: int = 20000,
    n_targeted: int = 0,
    n_archetype: int = 0,
    n_recipe: int = 0,
    profile_path: Optional[str] = None,
    seed: int = 0,
    shard_size: int = 2500,
    dilution_prob: float = 0.3,
    dilution_max_extra: int = 3,
    length_range=(96, 4096),
) -> dict:
    """Write the synth-v2 corpus (H1): ``synth_general`` + (profile-conditioned)
    ``synth_targeted``, both tagged into the shard index. ``synth_targeted`` requires
    a fitted ``--profile`` (``tetris.data.test_profile``). ``n_recipe`` writes the H1.1
    **targeted recipe** family — ``n_recipe`` series per GIFT-Eval config (tagged ``kind=config``)."""
    from .synthetic_v2 import write_general_corpus
    builder = dict(seed=seed, n_general=n_general, n_targeted=n_targeted,
                   n_archetype=n_archetype, n_recipe=n_recipe, shard_size=shard_size,
                   length_range=list(length_range),
                   dilution_prob=dilution_prob, profile=profile_path or "")
    with ShardWriter(out, shard_size=shard_size, builder=builder) as w:
        if n_archetype > 0:                              # H1.1 learnable-archetype family
            from .synth_archetype_recipes import write_archetype_corpus
            write_archetype_corpus(w, n_series=n_archetype, seed=seed,
                                   length_range=length_range)
        if n_recipe > 0:                                 # H1.1 targeted per-config recipe family
            from .synth_archetype_recipes import write_recipe_corpus
            write_recipe_corpus(w, n_per_config=n_recipe, seed=seed,
                                length_range=(max(512, length_range[0]), length_range[1]))
        if n_general > 0:
            write_general_corpus(w, n_series=n_general, seed=seed,
                                 length_range=length_range,
                                 dilution_prob=dilution_prob,
                                 dilution_max_extra=dilution_max_extra)
        if n_targeted > 0:
            if not profile_path:
                raise ValueError("--profile is required to materialize synth_targeted")
            from .synthetic_targeted import write_targeted_corpus
            from .test_profile import TestProfile
            write_targeted_corpus(w, n_series=n_targeted,
                                  profile=TestProfile.load(profile_path), seed=seed)
    return ShardReader(out).manifest


def main() -> None:  # pragma: no cover - manual entrypoint
    ap = argparse.ArgumentParser(description="Materialize a TETRIS streaming corpus")
    ap.add_argument("--out", required=True, help="output corpus directory")
    ap.add_argument("--n-synthetic", type=int, default=20000,
                    help="G4 varied-synthetic count (legacy 'synthetic' family)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=2500)
    ap.add_argument("--pretrain-root", default=None,
                    help="downloaded GiftEvalPretrain root (optional)")
    ap.add_argument("--pretrain-max-series", type=int, default=None)
    # synth-v2 (H1): general + targeted families. When either is >0 the v2 path runs.
    ap.add_argument("--n-general", type=int, default=0, help="synth_general count (H1)")
    ap.add_argument("--n-targeted", type=int, default=0, help="synth_targeted count (H1)")
    ap.add_argument("--n-archetype", type=int, default=0,
                    help="learnable-archetype count (H1.1; gen_variety cross-product)")
    ap.add_argument("--n-recipe", type=int, default=0,
                    help="targeted recipe count PER GIFT-Eval config (H1.1; data resembling the test split)")
    ap.add_argument("--profile", default=None, help="TestProfile JSON (needed for targeted)")
    ap.add_argument("--dilution-prob", type=float, default=0.3)
    ap.add_argument("--dilution-max-extra", type=int, default=3)
    args = ap.parse_args()

    if args.n_general or args.n_targeted or args.n_archetype or args.n_recipe:
        manifest = build_corpus_v2(
            args.out, n_general=args.n_general, n_targeted=args.n_targeted,
            n_archetype=args.n_archetype, n_recipe=args.n_recipe,
            profile_path=(os.path.expanduser(args.profile) if args.profile else None),
            seed=args.seed, shard_size=args.shard_size,
            dilution_prob=args.dilution_prob, dilution_max_extra=args.dilution_max_extra)
    else:
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
