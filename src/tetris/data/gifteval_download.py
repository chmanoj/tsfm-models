"""Real GIFT-Eval test-split download (O1, S12).

GIFT-Eval — *not* the Pretrain corpus — is publicly fetchable. The test data and
the 97-config table come from the HuggingFace dataset ``Salesforce/GiftEval``
(gluonts-interface arrow datasets); see ``SalesforceAIResearch/gift-eval``.

This module is the **only** seam that touches the network, and its heavyweight
deps (``huggingface_hub`` for the fetch, the official ``gift_eval`` package for the
gluonts split) are **lazy-imported, not project dependencies** — so the offline
test path (``eval_loader.make_synthetic_eval_shard``) and the rest of the package
never require them. Install the extras and call these only when you actually want
the real data::

    pip install huggingface_hub gift_eval
    python -c "from tetris.data.gifteval_download import download_gifteval; \
               download_gifteval('~/.cache/gifteval')"

Everything downstream consumes :class:`~tetris.data.contract.EvalItem`, so the
synthetic shard and the real download are interchangeable to the eval loop.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

import torch

from .contract import EvalItem

GIFTEVAL_REPO = "Salesforce/GiftEval"  # HF dataset repo id (verified, O1)


def download_gifteval(local_dir: str, *, configs: Optional[List[str]] = None) -> str:
    """Snapshot the real GIFT-Eval dataset to ``local_dir`` (lazy HF import).

    Returns the resolved local path. ``configs`` optionally restricts the fetch to
    a subset of dataset/freq subtrees (``allow_patterns``). Network + heavyweight;
    never invoked by tests.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError(
            "download_gifteval needs `huggingface_hub` (an optional extra, not a "
            "project dependency). Install it with `pip install huggingface_hub`."
        ) from e

    local_dir = os.path.expanduser(local_dir)
    allow = [f"{c}/*" for c in configs] if configs else None
    return snapshot_download(
        repo_id=GIFTEVAL_REPO,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow,
    )


def iter_eval_items(
    local_dir: str,
    *,
    configs: Optional[List[str]] = None,
    max_windows: int = 100,
) -> Iterator[EvalItem]:
    """Yield :class:`EvalItem`\\ s from a downloaded GIFT-Eval tree (lazy import).

    Uses the official ``gift_eval`` ``Dataset`` class to take the deterministic
    **first ``max_windows`` test windows per config** (D13), splitting each into a
    context-only ``data_tensor`` and the held-out ``y_true`` horizon. The
    seasonal-naive denominator is left ``None`` (MASE deferred, O4). Network-free
    given a populated ``local_dir``, but still depends on the optional ``gift_eval``
    package + the real arrow data, so it is not exercised in CI.
    """
    try:
        from gift_eval.data import Dataset  # type: ignore
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError(
            "iter_eval_items needs the official `gift_eval` package (optional "
            "extra). Install it from SalesforceAIResearch/gift-eval."
        ) from e

    local_dir = os.path.expanduser(local_dir)
    names = configs if configs is not None else _list_configs(local_dir)
    for config_id in names:  # pragma: no cover - requires real data
        ds = Dataset(name=config_id, term="short", to_univariate=False, storage_path=local_dir)
        for window_i, (ctx, future) in enumerate(zip(ds.test_data.input, ds.test_data.label)):
            if window_i >= max_windows:
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
            )


def _list_configs(local_dir: str) -> List[str]:  # pragma: no cover - requires real data
    """Enumerate ``dataset/freq`` config subtrees present in a downloaded tree."""
    out: List[str] = []
    for dataset in sorted(os.listdir(local_dir)):
        ddir = os.path.join(local_dir, dataset)
        if not os.path.isdir(ddir):
            continue
        for freq in sorted(os.listdir(ddir)):
            if os.path.isdir(os.path.join(ddir, freq)):
                out.append(f"{dataset}/{freq}")
    return out


def _target_2d(target):  # pragma: no cover - requires real data
    """gluonts target → channel-major 2-D ``[C, t]`` (univariate → ``[1, t]``)."""
    import numpy as np

    arr = np.asarray(target, dtype=np.float32)
    return arr[None, :] if arr.ndim == 1 else arr
