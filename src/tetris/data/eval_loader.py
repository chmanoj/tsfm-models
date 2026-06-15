"""GIFT-Eval test loader + record-only test loss (§6, S12).

The eval path scores the **held-out horizon** without forking the training data
path. An :class:`~tetris.data.contract.EvalItem` carries a context-only
``data_tensor`` plus the held-out ``y_true``; :func:`eval_batch` rebuilds a
*training-style* segment from ``cat(context, y_true)`` with the **forecast origin
forced at the context boundary** and ``p = len(y_true)``, then runs the
**unchanged** ``assemble`` → ``pack`` → horizon head. Because query slots are
``content_state == MASK`` and the causal mask blocks context→horizon, ``y_true``
only ever lands in ``horizon_target`` (never the observed store) — the same
no-leakage guarantee as training, with zero edits to the frozen S4/S6 tokenizer.

Test loss (D13) is the **record-only** horizon MAE on the deterministic first-``N``
windows per config; the inference variate basis is fixed (D4). MASE is deferred
(O4) — ``EvalItem.naive_denom`` stays ``None``.

The map-style :class:`GiftEvalEvalLoader` wraps a list of ``EvalItem``\\ s from
either the real download (:func:`from_download`, lazy/network) or an **offline
synthetic shard** (:func:`make_synthetic_eval_shard`, used by tests/shakedown).
``build_eval_loader`` (in ``contract``) keys between them.
"""

from __future__ import annotations

from math import ceil
from typing import List, Optional

import numpy as np
import torch

from .. import telescope as TS
from ..constants import ChannelRole, Role
from ..metrics import (
    horizon_test_loss,
    mase,
    seasonal_naive_denom,
    seasonal_naive_forecast,
)
from ..packing.collator import Batch, pack
from ..tokenize.assemble import assemble
from ..tokenize.spec import SegmentSpec
from ..tokenize.window_sampler import SamplerParams
from ..train.step import make_basis
from . import synthetic as S
from .contract import EvalItem, to_train_item, validate_item


# --- fixed-origin eval segment (mirrors window_sampler's budget math) ----------

def eval_spec(n_features: int, n_targets: int, t_ctx: int, p: int,
              params: SamplerParams) -> SegmentSpec:
    """A :class:`SegmentSpec` with origin pinned at the context end and fixed ``p``.

    Identical budget arithmetic to ``window_sampler.sample_window`` (per-channel
    token allowance, telescope coverage clipped to history) but **deterministic**:
    no random origin/horizon, no KFF reveal. Eval items pack one-per-buffer, so
    ``S ≤ L`` must hold — guaranteed while ``Q_total < L`` (modest benchmark ``p``).
    """
    C = n_features + n_targets
    channel_roles = [int(ChannelRole.FEATURE)] * n_features + [int(ChannelRole.TARGET)] * n_targets
    q_tok = ceil(p / params.p_out)
    Q = n_targets * q_tok
    Q_total = Q  # no KFF in eval
    n = max(1, (params.l_pack - Q_total) // C)
    origin = t_ctx  # forecast origin = context boundary (the held-out split point)

    prior = params.tier_prior
    t_cov = TS.coverage(TS.default_counts(n, prior))
    target_cov = min(t_cov, origin)
    n_eff = max(1, TS.tokens_for_coverage(target_cov, max_tokens=n, prior=prior))
    counts = TS.allocate(n_eff, target_cov, prior=prior)
    return SegmentSpec(
        C=C, n_features=n_features, n_targets=n_targets,
        origin=origin, p=p, channel_roles=channel_roles,
        Q=Q, K=0, Q_total=Q_total, n=n, counts=counts,
    )


def eval_batch(item: EvalItem, params: SamplerParams, *, p_out: int, l_pack: int) -> Batch:
    """Pack one :class:`EvalItem` into a one-segment buffer for scoring.

    The held-out ``y_true`` is appended after the context (target rows only;
    feature rows get NaN, which carry no horizon tokens) so ``assemble`` reads it
    as the post-origin horizon → ``horizon_target``. Context stats are computed
    from context only, so normalization never sees ``y_true``.
    """
    validate_item(to_train_item(item))
    data = item.data_tensor
    nf, nt = item.num_features, item.num_targets
    C, t_ctx = data.shape
    p = int(item.y_true.shape[0])
    assert item.y_true.shape[1] == nt, (item.y_true.shape, nt)

    full = torch.full((C, t_ctx + p), float("nan"), dtype=torch.float32)
    full[:, :t_ctx] = data
    full[nf:, t_ctx:] = item.y_true.to(torch.float32).T  # [p, nt] -> [nt, p]

    spec = eval_spec(nf, nt, t_ctx, p, params)
    seg = assemble((full, nf, nt), spec, p_out)
    return pack([[seg]], l_pack=l_pack, p_out=p_out)


# --- map-style loader ----------------------------------------------------------

class GiftEvalEvalLoader:
    """Map-style loader over a fixed list of :class:`EvalItem`\\ s (record-only)."""

    def __init__(self, items: List[EvalItem]) -> None:
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> EvalItem:
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    @classmethod
    def from_download(cls, cfg, *, local_dir: str, configs=None) -> "GiftEvalEvalLoader":
        """Real GIFT-Eval (O1; lazy/network — see ``gifteval_download``)."""
        from .gifteval_download import iter_eval_items

        items = list(iter_eval_items(local_dir, configs=configs,
                                     max_windows=cfg.eval.shard_windows))
        return cls(items)

    @classmethod
    def from_synthetic(cls, cfg, *, n_items: int, seed: int = 0) -> "GiftEvalEvalLoader":
        return cls(make_synthetic_eval_shard(cfg, n_items=n_items, seed=seed))


def make_synthetic_eval_shard(cfg, *, n_items: int, seed: int = 0) -> List[EvalItem]:
    """Offline deterministic eval shard mirroring the GIFT-Eval contract.

    Each item is a synthetic series split into a context ``[C, t_ctx]`` and a
    held-out horizon ``y_true [p, n_targets]``. Used by tests and the shakedown
    eval shard — no network, no optional deps."""
    rng = np.random.default_rng((seed, 0xE7A1))
    p_out = cfg.model.out_patch
    items: List[EvalItem] = []
    for i in range(n_items):
        rng_i = np.random.default_rng((seed, i))
        n = int(rng_i.integers(64, 257))
        p = int(p_out * rng_i.integers(1, 3))  # 1–2 horizon patches
        if rng_i.random() < 0.5:  # univariate
            x = S.gen_primitive(rng_i, n)[None, :]
            nf, nt = 0, 1
        else:                      # small multivariate (all-target)
            C = int(rng_i.integers(2, 4))
            x = S.gen_shared_factor(rng_i, n, C)
            nf, nt = 0, C
        x = np.ascontiguousarray(x, dtype=np.float32)
        context = torch.from_numpy(x[:, : n - p])
        y_true = torch.from_numpy(x[nf:, n - p :].T.copy())  # [p, nt]
        items.append(EvalItem(
            data_tensor=context, num_features=nf, num_targets=nt,
            y_true=y_true, naive_denom=None, config_id=f"synthetic/item_{i}",
        ))
    return items


# --- record-only test loss (D13) ----------------------------------------------

@torch.no_grad()
def evaluate_test_loss(
    forward,
    loader: GiftEvalEvalLoader,
    cfg,
    *,
    device: str = "cpu",
    max_windows: Optional[int] = None,
    basis_seed: int = 0,
) -> float:
    """Mean record-only horizon MAE over the first ``max_windows`` eval windows.

    One segment per buffer (eval is not packed); the inference variate basis is
    fixed by ``basis_seed`` (D4 deterministic at inference). Returns ``nan`` for an
    empty shard. Never updates parameters."""
    params = SamplerParams(
        l_pack=cfg.packing.L_pack, p_out=cfg.model.out_patch,
        tier_prior=tuple(cfg.model.tier_alloc_per_channel),
    )
    cap = cfg.eval.shard_windows if max_windows is None else max_windows
    total, count = 0.0, 0
    for i, item in enumerate(loader):
        if i >= cap:
            break
        batch = eval_batch(item, params, p_out=cfg.model.out_patch,
                           l_pack=cfg.packing.L_pack).to(device)
        gen = torch.Generator(device=device).manual_seed(basis_seed)
        basis = make_basis(batch, cfg.model.d_model, device=device, generator=gen)
        out = forward(batch, variate_basis=basis)
        total += horizon_test_loss(out, batch)
        count += 1
    return total / count if count else float("nan")


# --- raw-space forecast reconstruction + MASE (O4) ----------------------------

def horizon_forecast_raw(out, batch, item: EvalItem, *, p_out: int) -> torch.Tensor:
    """Invert the horizon head to **raw value space** for one eval item.

    Query tokens (``role == QRY``) carry the per-patch horizon prediction in
    anchored-arcsinh space; ``raw = sinh(pred)·σ + a`` per token (D10 Stage-9
    inversion, stats broadcast on the Batch). Each query token maps to target
    channel ``channel_idx - nf`` and horizon patch ``j`` (from ``t_center``);
    returns ``[p, num_targets]`` (NaN where no query covers a step)."""
    nf, nt = item.num_features, item.num_targets
    p = int(item.y_true.shape[0])
    pred = out.horizon[0]                         # [L, P_out] (B=1, one segment)
    role = batch.role[0]
    chan = batch.channel_idx[0]
    tcen = batch.t_center[0]
    a = batch.stats_a[0]
    sig = batch.stats_sigma[0]

    forecast = torch.full((p, nt), float("nan"), dtype=torch.float32, device=pred.device)
    qpos = (role == int(Role.QRY)).nonzero(as_tuple=True)[0]
    for pos in qpos.tolist():
        ti = int(chan[pos]) - nf
        if ti < 0 or ti >= nt:
            continue
        j = int(round((float(tcen[pos]) - p_out * 0.5) / p_out))
        raw = torch.sinh(pred[pos]) * sig[pos] + a[pos]   # [P_out]
        lo = j * p_out
        hi = min(lo + p_out, p)
        if lo >= p:
            continue
        forecast[lo:hi, ti] = raw[: hi - lo].to(torch.float32)
    return forecast


@torch.no_grad()
def evaluate_mase(
    forward,
    loader: GiftEvalEvalLoader,
    cfg,
    *,
    device: str = "cpu",
    max_windows: Optional[int] = None,
    basis_seed: int = 0,
):
    """MASE of the model vs the seasonal-naive baseline (GIFT-Eval style, O4).

    For each eval item the model's horizon is inverted to raw space and scored
    against the held-out ``y_true`` with the **dataset-provided** ``season_length``
    (``EvalItem.season_length``; never detected here). Per target channel:
    ``MASE = MAE_horizon / in-sample seasonal-naive denom``. Aggregates as the
    geometric mean across all channel-items (leaderboard convention). Returns a
    dict with model/seasonal-naive gmeans and the skill ratio (``<1`` beats naive).
    """
    params = SamplerParams(
        l_pack=cfg.packing.L_pack, p_out=cfg.model.out_patch,
        tier_prior=tuple(cfg.model.tier_alloc_per_channel),
    )
    p_out = cfg.model.out_patch
    cap = cfg.eval.shard_windows if max_windows is None else max_windows
    log_model: List[float] = []
    log_snaive: List[float] = []
    skipped = 0
    for i, item in enumerate(loader):
        if i >= cap:
            break
        if item.season_length is None:
            skipped += 1
            continue
        nf, nt = item.num_features, item.num_targets
        batch = eval_batch(item, params, p_out=p_out, l_pack=cfg.packing.L_pack).to(device)
        gen = torch.Generator(device=device).manual_seed(basis_seed)
        basis = make_basis(batch, cfg.model.d_model, device=device, generator=gen)
        out = forward(batch, variate_basis=basis)
        forecast = horizon_forecast_raw(out, batch, item, p_out=p_out).cpu()  # [p, nt]
        y_true = item.y_true.to(torch.float32)                                # [p, nt]
        context = item.data_tensor                                           # [C, t_ctx]
        for ti in range(nt):
            # per-channel seasonality (multi-freq sanity); falls back to series m
            m = int(item.channel_seasons[nf + ti]) if item.channel_seasons else int(item.season_length)
            ctx_c = context[nf + ti]
            denom = seasonal_naive_denom(ctx_c, m)
            yt = y_true[:, ti]
            fc = forecast[:, ti]
            valid = torch.isfinite(fc) & torch.isfinite(yt)
            if not bool(valid.any()):
                continue
            snaive = seasonal_naive_forecast(ctx_c, m, yt.shape[0])
            log_model.append(mase(yt[valid], fc[valid], denom))
            log_snaive.append(mase(yt[valid], snaive[valid], denom))

    def _gmean(xs):
        if not xs:
            return float("nan")
        t = torch.tensor(xs, dtype=torch.float64).clamp_min(1e-12)
        return float(torch.exp(t.log().mean()))

    model_g, snaive_g = _gmean(log_model), _gmean(log_snaive)
    return {
        "model_mase": model_g,
        "snaive_mase": snaive_g,
        "skill": (model_g / snaive_g) if snaive_g and snaive_g == snaive_g else float("nan"),
        "n": len(log_model),
        "skipped": skipped,
    }
