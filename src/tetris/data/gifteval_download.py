"""Real GIFT-Eval download + test/train iterators (O1, S12, G2).

GIFT-Eval — *not* the Pretrain corpus — is publicly fetchable. The test data and
the 97-config table come from the HuggingFace dataset ``Salesforce/GiftEval``
(gluonts-interface arrow datasets); see ``SalesforceAIResearch/gift-eval``.

This module is the **only** seam that touches the network, and its heavyweight
deps (``huggingface_hub`` for the fetch, the official ``gift_eval`` package +
``gluonts`` for the split/seasonality) are **lazy-imported, not project
dependencies** — so the offline test path (``eval_loader.make_synthetic_eval_shard``)
and the rest of the package never require them. Install the extras and call these
only when you actually want the real data::

    pip install huggingface_hub gluonts \
        "git+https://github.com/SalesforceAIResearch/gift-eval.git"
    export GIFT_EVAL=~/Projects/gifteval/test     # storage root (the package reads this)
    python -c "from tetris.data.gifteval_download import download_gifteval; \
               download_gifteval('~/Projects/gifteval/test')"

Everything downstream consumes :class:`~tetris.data.contract.EvalItem` (test) or
the frozen training ``Item`` 3-tuple (train), so the synthetic shard and the real
download are interchangeable to the eval/train loops.

**G2 API re-verification (against the installed ``gift_eval``):** the ``Dataset``
class takes ``(name, term, to_univariate, storage_env_var="GIFT_EVAL")`` — it reads
the storage root from the **``GIFT_EVAL`` env var**, *not* a ``storage_path`` kwarg
(the S12 guess was wrong and would ``TypeError``). ``term`` is ``short|medium|long``
(horizon ×1/10/15). It exposes ``.test_data.{input,label}``, ``.training_dataset``,
``.prediction_length``, ``.windows``, and ``.freq``. Seasonality for MASE is **not**
on the package — GIFT-Eval's own scoring derives it from the frequency via gluonts'
``get_seasonality(freq)`` (e.g. H→24, M→12, D→1, W→1, T→1440), so that is the
canonical, non-silent source we wire into ``EvalItem.season_length``.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

import torch

from .contract import Item, EvalItem, validate_item

GIFTEVAL_REPO = "Salesforce/GiftEval"  # HF dataset repo id (verified, O1)

# Storage roots are resolved from env vars (maintainer's G2 choice: paths must be
# available to the training/eval code without hardcoding). GIFT_EVAL is *also* the
# var the official ``gift_eval`` package reads, so a single export wires both.
TEST_ENV_VAR = "GIFT_EVAL"               # test-split root, e.g. ~/Projects/gifteval/test
PRETRAIN_ENV_VAR = "GIFT_EVAL_PRETRAIN"  # pretrain root (reserved for G5)

_HF_DATASET_MARKER = "dataset_info.json"  # marks a load_from_disk leaf directory


def download_gifteval(local_dir: str = "", *, configs: Optional[List[str]] = None) -> str:
    """Snapshot the real GIFT-Eval dataset to ``local_dir`` (lazy HF import).

    ``local_dir`` falls back to ``$GIFT_EVAL`` when empty. Returns the resolved
    local path. ``configs`` optionally restricts the fetch to a subset of
    dataset/freq subtrees (``allow_patterns``). Network + heavyweight; never
    invoked by tests.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError(
            "download_gifteval needs `huggingface_hub` (an optional extra, not a "
            "project dependency). Install it with `pip install huggingface_hub`."
        ) from e

    local_dir = _resolve_local_dir(local_dir)
    allow = [f"{c}/*" for c in configs] if configs else None
    return snapshot_download(
        repo_id=GIFTEVAL_REPO,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow,
    )


def _resolve_local_dir(local_dir: str, *, env_var: str = TEST_ENV_VAR) -> str:
    """Resolve a storage root: explicit ``local_dir`` wins, else ``$env_var``.

    Loads the repo-root ``.env`` first (the same mechanism ``gift_eval`` uses, so a
    single ``GIFT_EVAL=…`` line makes the path available to both us and the package
    without exporting it) — lazy/optional, never overrides an already-set var.
    """
    if not local_dir and env_var not in os.environ:
        try:  # lazy + optional: python-dotenv ships with the gift_eval extras
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover - optional extra
            pass
    resolved = local_dir or os.getenv(env_var, "")
    if not resolved:
        raise ValueError(
            f"no GIFT-Eval storage root: pass local_dir, set ${env_var}, or add "
            f"{env_var}=… to the repo-root .env (e.g. {env_var}=~/Projects/gifteval/test)."
        )
    return os.path.expanduser(resolved)


def _cap(n: int) -> int:
    """Per-config cap; ``-1`` -> unbounded (a huge sentinel)."""
    return (1 << 60) if n is None or n < 0 else int(n)


def _make_dataset(name: str, term: str, local_dir: str, *, to_univariate: bool = False):
    """Construct a ``gift_eval.data.Dataset`` rooted at ``local_dir`` (lazy import).

    The package reads the storage root from ``$GIFT_EVAL`` (its ``storage_env_var``
    default), so we point that at ``local_dir`` for the call.
    """
    try:
        from gift_eval.data import Dataset  # type: ignore
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError(
            "iter_eval_items/iter_train_items need the official `gift_eval` package "
            "(optional extra, + gluonts). Install from SalesforceAIResearch/gift-eval."
        ) from e

    os.environ[TEST_ENV_VAR] = local_dir  # what Dataset(storage_env_var=...) reads
    return Dataset(name=name, term=term, to_univariate=to_univariate)


def _season_length(freq: str) -> int:
    """Dataset seasonality m for MASE — GIFT-Eval's own source (gluonts).

    GIFT-Eval scores seasonal-naive/MASE with gluonts' ``get_seasonality(freq)``
    (H→24, M→12, D→1, W→1, T→1440, …). No silent fallback: an unknown freq is
    gluonts' responsibility (it returns 1 with a warning), surfaced here verbatim.
    """
    from gluonts.time_feature import get_seasonality  # lazy (gift_eval dep)

    return int(get_seasonality(freq))


def iter_eval_items(
    local_dir: str = "",
    *,
    configs: Optional[List[str]] = None,
    term: str = "short",
    items_per_config: int = 10,
    to_univariate: bool = False,
) -> Iterator[EvalItem]:
    """Yield :class:`EvalItem`\\ s from a downloaded GIFT-Eval tree (lazy import).

    The deterministic **first ``items_per_config`` test windows per config** (D13;
    ``-1`` -> all), each split into a context-only ``data_tensor`` and the held-out
    ``y_true`` horizon. ``season_length`` is set from the dataset frequency via
    GIFT-Eval's own ``get_seasonality`` so MASE is correct; ``config_id`` is
    ``"{name}/{term}"`` so the leaderboard distinguishes terms. Network-free given a
    populated ``local_dir`` (or ``$GIFT_EVAL``), but depends on the optional
    ``gift_eval``/``gluonts`` packages + real arrow data, so it is not run in CI.
    """
    local_dir = _resolve_local_dir(local_dir)
    names = configs if configs is not None else _list_configs(local_dir)
    cap = _cap(items_per_config)
    for name in names:  # pragma: no cover - requires real data
        ds = _make_dataset(name, term, local_dir, to_univariate=to_univariate)
        season = _season_length(ds.freq)
        config_id = f"{name}/{term}"
        for window_i, (ctx, future) in enumerate(zip(ds.test_data.input, ds.test_data.label)):
            if window_i >= cap:
                break
            context = torch.as_tensor(_target_2d(ctx["target"]), dtype=torch.float32)
            y_true = torch.as_tensor(_target_2d(future["target"]).T, dtype=torch.float32)
            n_targets = context.shape[0]
            yield EvalItem(
                data_tensor=context,
                num_features=0,
                num_targets=n_targets,
                y_true=y_true,
                naive_denom=None,
                config_id=config_id,
                season_length=season,
            )


def iter_train_items(
    local_dir: str = "",
    *,
    configs: Optional[List[str]] = None,
    term: str = "short",
    max_series_per_config: int = -1,
    to_univariate: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[Item]:
    """Yield frozen training ``Item``\\ s from GIFT-Eval **train** series (lazy import).

    The ``train`` split is everything before the test/validation windows
    (``Dataset.training_dataset``), not the rolling test windows. Each gluonts series
    becomes a frozen contract tuple ``(data [C,t] float32, num_features=0,
    num_targets=C)``, features-first, raw, NaNs allowed (kept as-is). Series are
    sharded disjointly by ``(rank, world_size)`` over the flattened stream (§9; no-op
    at world_size=1). ``max_series_per_config`` caps series per config (``-1`` -> all).
    G2 builds this reader but does **not** wire a training factory key yet.
    """
    local_dir = _resolve_local_dir(local_dir)
    names = configs if configs is not None else _list_configs(local_dir)
    cap = _cap(max_series_per_config)
    gi = 0  # global series index for round-robin rank sharding
    for name in names:  # pragma: no cover - requires real data
        ds = _make_dataset(name, term, local_dir, to_univariate=to_univariate)
        for series_i, entry in enumerate(ds.training_dataset):
            if series_i >= cap:
                break
            take = (gi % world_size) == rank
            gi += 1
            if not take:
                continue
            data = torch.as_tensor(_target_2d(entry["target"]), dtype=torch.float32)
            nt = data.shape[0]
            item: Item = (data, 0, nt)
            validate_item(item)
            yield item


def list_configs(local_dir: str = "") -> List[str]:
    """Public: enumerate the GIFT-Eval config names present under the storage root."""
    return _list_configs(_resolve_local_dir(local_dir))


def _list_configs(local_dir: str) -> List[str]:  # pragma: no cover - requires real data
    """Enumerate config names (``Dataset(name=...)``) in a downloaded tree.

    A config is a ``load_from_disk`` leaf — a directory containing
    ``dataset_info.json`` — and its name is the path **relative to the storage
    root** (e.g. ``electricity/15T``, ``m4_yearly``). Robust to variable nesting
    depth across the 97 configs.
    """
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(local_dir):
        if _HF_DATASET_MARKER in filenames:
            rel = os.path.relpath(dirpath, local_dir)
            if rel != ".":
                out.append(rel)
    return sorted(out)


def _target_2d(target):  # pragma: no cover - requires real data
    """gluonts target → channel-major 2-D ``[C, t]`` (univariate → ``[1, t]``)."""
    import numpy as np

    arr = np.asarray(target, dtype=np.float32)
    return arr[None, :] if arr.ndim == 1 else arr
