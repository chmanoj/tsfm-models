"""G3 — GIFT-Eval ``test``-split overfit loader (test-as-training).

All CI-safe / offline: the loader's cycle + rank-sharding + Item-only contract are
exercised on a hand-built ``EvalItem`` shard (no ``gift_eval``/network), and the
``build_loader`` factory key is checked by monkeypatching ``iter_eval_items`` so the
real-data path is never touched in CI. The real ``from_cfg`` (lazy ``gift_eval`` +
download) is covered by the manual WSL run, not here.
"""

from __future__ import annotations

import os
from itertools import islice

import numpy as np
import pytest
import torch

from tetris.config import load_config
from tetris.data.contract import EvalItem, validate_item
from tetris.data import gifteval_download as gd
from tetris.data.gifteval_overfit_loader import (
    GiftEvalTestOverfitLoader,
    merge_context_horizon,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _eval_items(n_configs=3, per_config=4):
    """A small multi-config EvalItem shard with context-only data_tensor + held-out
    horizon (the held-out fields must NOT leak through to the training Item)."""
    items = []
    for c in range(n_configs):
        for w in range(per_config):
            t_ctx = 32 + c  # varied lengths
            x = np.sin((np.arange(t_ctx + 8) + 3 * c + w) * 0.3).astype(np.float32)[None, :]
            items.append(EvalItem(
                data_tensor=torch.from_numpy(x[:, :t_ctx].copy()),
                num_features=0, num_targets=1,
                y_true=torch.from_numpy(x[:, t_ctx:].T.copy()),
                naive_denom=None, config_id=f"cfg/{c}/short", season_length=8,
            ))
    return items


def test_overfit_loader_merges_context_and_horizon():
    """Each yielded item is a frozen 3-tuple Item that is the **continuous**
    cat(context, horizon) series (G3.1 overfit) — the held-out y_true is now part of
    the training series so the model can overfit it."""
    eval_items = _eval_items()
    loader = GiftEvalTestOverfitLoader.from_eval_items(eval_items)
    got = list(islice(iter(loader), len(eval_items)))
    assert len(got) == len(eval_items)
    for item, e in zip(got, eval_items):
        assert isinstance(item, tuple) and len(item) == 3  # not an EvalItem (NamedTuple len 9)
        validate_item(item)
        t_ctx, p = e.data_tensor.shape[1], e.y_true.shape[0]
        assert item[0].shape == (e.num_features + e.num_targets, t_ctx + p)  # merged length
        assert torch.equal(item[0][:, :t_ctx], e.data_tensor)               # context prefix
        # target rows' tail == the held-out horizon (now trained on, by design)
        assert torch.equal(item[0][e.num_features:, t_ctx:], e.y_true.T)


def test_overfit_loader_cycles():
    """The finite pool cycles (overfit) — more items than the pool, wrapping in order."""
    eval_items = _eval_items(n_configs=2, per_config=3)  # pool of 6
    loader = GiftEvalTestOverfitLoader.from_eval_items(eval_items)
    got = list(islice(iter(loader), len(eval_items) * 2 + 1))  # 13 from a pool of 6
    assert len(got) == 13
    assert torch.equal(got[0][0], got[6][0])   # wrapped around
    assert torch.equal(got[1][0], got[7][0])
    assert len(loader) == 6


def test_overfit_loader_no_cycle_drains():
    eval_items = _eval_items(n_configs=1, per_config=4)
    loader = GiftEvalTestOverfitLoader.from_eval_items(eval_items, cycle=False)
    assert len(list(iter(loader))) == 4  # stops after one pass


def test_overfit_loader_rank_shard_disjoint_and_complete():
    """world_size=2 shards round-robin: disjoint, union == full pool (O6)."""
    eval_items = _eval_items(n_configs=2, per_config=3)  # pool of 6
    r0 = GiftEvalTestOverfitLoader.from_eval_items(eval_items, rank=0, world_size=2)
    r1 = GiftEvalTestOverfitLoader.from_eval_items(eval_items, rank=1, world_size=2)
    assert len(r0) == 3 and len(r1) == 3
    s0 = [tuple(it[0].flatten().tolist()) for it in islice(iter(r0), 3)]
    s1 = [tuple(it[0].flatten().tolist()) for it in islice(iter(r1), 3)]
    assert set(s0).isdisjoint(s1)
    full = {tuple(merge_context_horizon(e)[0].flatten().tolist()) for e in eval_items}
    assert set(s0) | set(s1) == full


def test_overfit_loader_empty_shard_raises():
    """A rank with no items fails loudly rather than spinning forever."""
    eval_items = _eval_items(n_configs=1, per_config=1)  # pool of 1
    loader = GiftEvalTestOverfitLoader.from_eval_items(eval_items, rank=1, world_size=2)
    assert len(loader) == 0
    with pytest.raises(ValueError, match="no items"):
        next(iter(loader))


def test_overfit_loader_drops_too_short_series():
    """Merged series shorter than ``min_context`` are dropped (the sampler's floor);
    short-but-viable series (incl. ``t_raw < out_patch``) are kept (G3.1)."""
    def _mk(t_ctx, p, cid):
        x = np.zeros((1, t_ctx + p), np.float32)
        return EvalItem(torch.from_numpy(x[:, :t_ctx].copy()), 0, 1,
                        torch.from_numpy(x[:, t_ctx:].T.copy()), None, cid, season_length=4)

    # merged length = t_ctx + p
    eval_items = [_mk(3, 2, "a/short"),     # merged 5  -> dropped at min_context=10
                  _mk(20, 4, "b/short")]    # merged 24 -> kept
    loader = GiftEvalTestOverfitLoader.from_eval_items(eval_items, min_context=10)
    assert len(loader) == 1
    assert loader.items[0][0].shape[1] == 24
    # default floor (2) keeps a tiny 2 ctx + 2 pred series (incomplete patch OK)
    keep = GiftEvalTestOverfitLoader.from_eval_items([_mk(2, 2, "c/short")])
    assert len(keep) == 1 and keep.items[0][0].shape[1] == 4


def test_build_loader_factory_key_offline(monkeypatch):
    """The ``gifteval_test_overfit`` factory key dispatches to the loader and threads
    cfg.data.terms + items_per_config into iter_eval_items — without touching real data."""
    from tetris.data.contract import build_loader

    cfg = load_config(os.path.join(CONFIG_DIR, "gifteval_test_overfit.yaml"))
    captured = {}

    def fake_iter_eval_items(local_dir="", *, terms=("short",), items_per_config=10, **kw):
        captured["terms"] = tuple(terms)
        captured["items_per_config"] = items_per_config
        return iter(_eval_items(n_configs=2, per_config=2))

    monkeypatch.setattr(gd, "iter_eval_items", fake_iter_eval_items)
    loader = build_loader(cfg)
    assert isinstance(loader, GiftEvalTestOverfitLoader)
    assert captured["terms"] == tuple(cfg.data.terms)
    assert captured["items_per_config"] == cfg.eval.items_per_config
    first = next(iter(loader))
    validate_item(first)
