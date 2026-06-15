"""Loader contract (O2) + the ``build_loader`` factory.

The training item is the frozen 3-tuple

    Item = (data_tensor: torch.float32 [n_features + n_targets, t],
            num_features: int, num_targets: int)

features-first (the first ``num_features`` rows are feature channels, the rest
targets), raw values, may contain NaN. ``t`` and ``n = num_features+num_targets``
vary per item. The collator and model consume *only* this tuple, so the real
loaders drop in behind ``build_loader`` with zero downstream change.

``EvalItem`` (§6) is a superset whose first three fields *are* the training Item,
so a trivial adapter feeds the shared collator; held-out fields ride alongside
and are stripped before packing (no leakage, no training-path fork).
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple, Optional, Protocol, Tuple, runtime_checkable

import torch

# The frozen training contract.
Item = Tuple[torch.Tensor, int, int]


class EvalItem(NamedTuple):
    """GIFT-Eval test item (record-only scoring; D13/§6). Fields 1..3 are the
    training Item; held-out fields are stripped before packing."""

    data_tensor: torch.Tensor              # context only, [n, t_ctx], raw, may contain NaN
    num_features: int
    num_targets: int
    y_true: torch.Tensor                   # held-out horizon, [p, num_targets]
    naive_denom: Optional[torch.Tensor]    # seasonal-naive denominator; deferred (O4) -> None in v1
    config_id: str                         # which of the 97 GIFT-Eval configs
    season_length: Optional[int] = None    # dataset-provided seasonality m for MASE (O4); None -> unknown
    channel_seasons: Optional[List[int]] = None  # per-channel m (multi-freq sanity); falls back to season_length


def to_train_item(e: EvalItem) -> Item:
    """Strip an EvalItem to the training contract for the shared collator."""
    return (e.data_tensor, e.num_features, e.num_targets)


@runtime_checkable
class Loader(Protocol):
    """Anything iterable yielding training Items. The real Pretrain loader is
    ``__iter__``-style; the GIFT-Eval loader is map-style and adapted via
    ``to_train_item`` (S12)."""

    def __iter__(self) -> Iterator[Item]: ...


def validate_item(item: Item) -> None:
    """Assert an item honors the contract exactly. Cheap guard for tests and
    for the loader boundary."""
    if not (isinstance(item, tuple) and len(item) == 3):
        raise TypeError(f"Item must be a 3-tuple, got {type(item)} len={len(item) if hasattr(item, '__len__') else '?'}")
    data, nf, nt = item
    if not isinstance(data, torch.Tensor):
        raise TypeError("data_tensor must be a torch.Tensor")
    if data.dtype != torch.float32:
        raise TypeError(f"data_tensor must be float32, got {data.dtype}")
    if data.dim() != 2:
        raise ValueError(f"data_tensor must be 2-D [n, t], got shape {tuple(data.shape)}")
    if not (isinstance(nf, int) and isinstance(nt, int)):
        raise TypeError("num_features and num_targets must be ints")
    if nf < 0 or nt < 1:
        raise ValueError(f"need num_features >= 0 and num_targets >= 1, got {nf}, {nt}")
    if data.shape[0] != nf + nt:
        raise ValueError(f"rows {data.shape[0]} != num_features+num_targets {nf + nt}")


def build_loader(cfg, *, rank: int = 0, world_size: int = 1):
    """Factory keyed by ``cfg.data.loader``. Shards source series disjointly by
    rank (§9 distributed seam — no-op at world_size=1). Only the synthetic
    stand-in is wired in v1; ``gifteval_test`` lands at S12."""
    name = cfg.data.loader
    if name == "standin_pretrain":
        from .standin_loader import StandInPretrainLoader

        return StandInPretrainLoader.from_cfg(cfg, rank=rank, world_size=world_size)
    if name == "sanity":
        from .sanity_loader import SanityTrainLoader

        return SanityTrainLoader.from_cfg(cfg, rank=rank, world_size=world_size)
    raise NotImplementedError(
        f"training loader {name!r} unknown; GIFT-Eval is record-only — use build_eval_loader"
    )


def build_eval_loader(cfg, *, local_dir: Optional[str] = None):
    """Factory for the record-only eval loader (§6, S12), keyed by ``cfg.eval.loader``.

    ``gifteval_test`` is the real GIFT-Eval download (O1; lazy/network — needs
    ``local_dir``); ``synthetic_eval`` is the offline synthetic shard used by tests
    and the shakedown. Both yield :class:`EvalItem`; ``EvalItem`` never enters the
    training ``build_loader``/reservoir path (it is stripped via ``to_train_item``)."""
    from .eval_loader import GiftEvalEvalLoader

    name = cfg.eval.loader
    if name == "gifteval_test":
        if local_dir is None:
            raise ValueError("gifteval_test needs local_dir (run gifteval_download first)")
        return GiftEvalEvalLoader.from_download(cfg, local_dir=local_dir)
    if name == "synthetic_eval":
        return GiftEvalEvalLoader.from_synthetic(cfg, n_items=cfg.eval.shard_windows, seed=cfg.run.seed)
    if name == "sanity_eval":
        from .sanity_loader import make_sanity_eval_shard

        return GiftEvalEvalLoader(make_sanity_eval_shard(cfg))
    raise NotImplementedError(
        f"eval loader {name!r} unknown (use gifteval_test | synthetic_eval | sanity_eval)"
    )
