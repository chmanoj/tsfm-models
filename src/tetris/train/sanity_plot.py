"""Plot eval-sample forecasts — actual vs model vs seasonal naive (all cases).

Picks ``n`` random eval items and, for **every target channel** of each, overlays
the tail of the context, the held-out ground truth, the model forecast (horizon
head inverted to raw space), and the seasonal-naive baseline scored with that
channel's own period. Grid = samples (rows) × target channels (cols); works for
every sanity case (univariate or multivariate; each subplot reports its own ``m``).
"""

from __future__ import annotations

import random

import torch

from ..data.eval_loader import rollout_forecast
from ..metrics import seasonal_naive_forecast
from ..tokenize.window_sampler import SamplerParams


def _season_of(item, ch_global: int) -> int:
    if item.channel_seasons:
        return int(item.channel_seasons[ch_global])
    return int(item.season_length) if item.season_length else 1


@torch.no_grad()
def plot_eval_samples(forward, eval_loader, cfg, *, out_path: str, n: int = 3,
                      seed: int = 0, device: str = "cpu", basis_seed: int = 0) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(eval_loader)
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(items)), min(n, len(items))))
    params = SamplerParams(
        l_pack=cfg.packing.L_pack, p_out=cfg.model.out_patch,
        tier_prior=tuple(cfg.model.tier_alloc_per_channel),
        max_query_tokens=cfg.packing.max_query_tokens,
    )
    p_out = cfg.model.out_patch
    nt_max = max(items[i].num_targets for i in idxs)

    def _forecast(it):
        # Full-horizon forecast via the same iterative rollout the leaderboard uses
        # (G3.1): one capped pass for small horizons, ceil(p/p_pred) for long ones.
        return rollout_forecast(forward, it, params, d_model=cfg.model.d_model,
                                p_out=p_out, l_pack=cfg.packing.L_pack,
                                device=device, basis_seed=basis_seed)  # [p, nt]

    fig, axes = plt.subplots(len(idxs), nt_max, squeeze=False,
                             figsize=(max(11.0, 5.0 * nt_max), 2.7 * len(idxs) + 0.6),
                             constrained_layout=True)
    for row, i in enumerate(idxs):
        item = items[i]
        nf, nt = item.num_features, item.num_targets
        forecast = _forecast(item)
        # KFF cases: a counterfactual forecast with the feature future hidden, to
        # show the model fails without the known-future covariate (past-only).
        forecast_po = None
        if item.feature_future is not None:
            forecast_po = _forecast(item._replace(feature_future=None))

        for ti in range(nt_max):
            ax = axes[row][ti]
            if ti >= nt:
                ax.axis("off")
                continue
            ch = nf + ti
            m = _season_of(item, ch)
            ctx = item.data_tensor[ch].cpu()
            y_true = item.y_true[:, ti].cpu()
            snaive = seasonal_naive_forecast(item.data_tensor[ch], m, y_true.shape[0]).cpu()
            t_ctx, p = ctx.shape[0], y_true.shape[0]
            show = min(t_ctx, max(3 * m, 96))
            xs_ctx = range(t_ctx - show, t_ctx)
            xs_h = range(t_ctx, t_ctx + p)

            ax.plot(xs_ctx, ctx[-show:], color="0.6", lw=1, label="context")
            ax.plot(xs_h, y_true, color="black", lw=2, label="actual")
            model_label = "model (KFF)" if forecast_po is not None else "model"
            ax.plot(xs_h, forecast[:, ti], color="tab:blue", marker=".", label=model_label)
            if forecast_po is not None:
                ax.plot(xs_h, forecast_po[:, ti], color="tab:green", ls=":", marker="x",
                        ms=4, label="model (past-only)")
            ax.plot(xs_h, snaive, color="tab:orange", ls="--", label="seasonal_naive")
            ax.axvline(t_ctx - 0.5, color="red", alpha=0.3, lw=1)
            ax.set_title(f"series {i} · ch{ch} (m={m})", fontsize=8)
            if row == 0 and ti == 0:
                ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(f"{cfg.run.name} — actual vs model vs seasonal-naive", fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
