"""G5 — live GIFT-Eval ``train``-split loader (``gifteval_train`` factory key).

CI-safe: ``iter_train_items`` (the only lazy/network seam) is monkeypatched, so the
loader's cycle / threading / term-defaulting / empty-shard contract is exercised
without ``gift_eval`` or a download. The real ``from_cfg`` path is covered by the
manual WSL run.
"""

from __future__ import annotations

import os
from itertools import islice

import numpy as np
import pytest
import torch

from tetris.config import load_config
from tetris.data import gifteval_download as gd
from tetris.data.contract import validate_item
from tetris.data.gifteval_train_loader import GiftEvalTrainLoader

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _fake_train_items(n=4):
    """A deterministic pool of frozen Items standing in for the train split."""
    def gen(local_dir="", *, configs=None, term="short",
            max_series_per_config=-1, rank=0, world_size=1, **kw):
        gen.calls.append(dict(local_dir=local_dir, term=term,
                              max_series_per_config=max_series_per_config,
                              rank=rank, world_size=world_size))
        idx = 0
        for i in range(n):
            take = (idx % world_size) == rank
            idx += 1
            if not take:
                continue
            x = np.full((1, 16 + i), float(i), dtype=np.float32)
            item = (torch.from_numpy(x), 0, 1)
            validate_item(item)
            yield item
    gen.calls = []
    return gen


def test_train_loader_cycles_finite_pool(monkeypatch):
    monkeypatch.setattr(gd, "iter_train_items", _fake_train_items(n=3))
    loader = GiftEvalTrainLoader(term="short")
    got = list(islice(iter(loader), 7))             # pool of 3, cycled
    assert len(got) == 7
    assert torch.equal(got[0][0], got[3][0])        # wrapped in order
    assert torch.equal(got[1][0], got[4][0])


def test_train_loader_no_cycle_drains(monkeypatch):
    monkeypatch.setattr(gd, "iter_train_items", _fake_train_items(n=4))
    loader = GiftEvalTrainLoader(cycle=False)
    assert len(list(iter(loader))) == 4             # one pass, then stop


def test_train_loader_threads_rank_term_and_cap(monkeypatch):
    fake = _fake_train_items(n=6)
    monkeypatch.setattr(gd, "iter_train_items", fake)
    loader = GiftEvalTrainLoader(local_dir="/x", term="medium",
                                 max_series_per_config=5, rank=1, world_size=2)
    list(islice(iter(loader), 3))
    call = fake.calls[0]
    assert call == dict(local_dir="/x", term="medium",
                        max_series_per_config=5, rank=1, world_size=2)


def test_train_loader_rank_shard_disjoint_and_complete(monkeypatch):
    monkeypatch.setattr(gd, "iter_train_items", _fake_train_items(n=6))
    r0 = GiftEvalTrainLoader(rank=0, world_size=2)
    r1 = GiftEvalTrainLoader(rank=1, world_size=2)
    s0 = [tuple(it[0].flatten().tolist()) for it in islice(iter(r0), 3)]
    s1 = [tuple(it[0].flatten().tolist()) for it in islice(iter(r1), 3)]
    assert set(s0).isdisjoint(s1)
    monkeypatch.setattr(gd, "iter_train_items", _fake_train_items(n=6))
    full = {tuple(it[0].flatten().tolist())
            for it in islice(iter(GiftEvalTrainLoader(cycle=False)), 6)}
    assert set(s0) | set(s1) == full


def test_train_loader_empty_yield_raises(monkeypatch):
    monkeypatch.setattr(gd, "iter_train_items", _fake_train_items(n=0))
    with pytest.raises(ValueError, match="no train series"):
        next(iter(GiftEvalTrainLoader()))


def test_train_loader_from_cfg_term_defaults_to_first_term(monkeypatch):
    fake = _fake_train_items(n=2)
    monkeypatch.setattr(gd, "iter_train_items", fake)
    cfg = load_config(os.path.join(CONFIG_DIR, "gifteval_test_overfit.yaml"))
    # default train_term empty -> first of cfg.data.terms
    loader = GiftEvalTrainLoader.from_cfg(cfg)
    assert loader.term == cfg.data.terms[0]
    next(iter(loader))
    assert fake.calls[0]["term"] == cfg.data.terms[0]


def test_build_loader_factory_key_offline(monkeypatch):
    from tetris.data.contract import build_loader

    fake = _fake_train_items(n=3)
    monkeypatch.setattr(gd, "iter_train_items", fake)
    cfg = load_config(os.path.join(CONFIG_DIR, "gifteval_test_overfit.yaml"))
    cfg.data.loader = "gifteval_train"
    cfg.data.train_term = "long"
    cfg.data.train_max_series_per_config = 7
    loader = build_loader(cfg)
    assert isinstance(loader, GiftEvalTrainLoader)
    first = next(iter(loader))
    validate_item(first)
    assert fake.calls[0]["term"] == "long"
    assert fake.calls[0]["max_series_per_config"] == 7
