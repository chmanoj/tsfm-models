"""Plot eval-sample forecasts — actual vs model vs seasonal naive (all cases).

Picks ``n`` random eval items and overlays, for the first target channel, the tail
of the context, the held-out ground truth, the model forecast (horizon head
inverted to raw space), and the seasonal-naive baseline. Works for every sanity
case (univariate or multivariate; the title reports ``C`` and the season length).
"""

from __future__ import annotations

import random

import torch

from ..data.eval_loader import eval_batch, horizon_forecast_raw
from ..metrics import seasonal_naive_forecast
from ..tokenize.window_sampler import SamplerParams
from .step import make_basis


@torch.no_grad()
def plot_eval_samples(forward, eval_loader, cfg, *, out_path: str, n: int = 5,
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
    )
    p_out = cfg.model.out_patch

    fig, axes = plt.subplots(len(idxs), 1, figsize=(11, 2.6 * len(idxs)), squeeze=False)
    for row, i in enumerate(idxs):
        item = items[i]
        nf, nt = item.num_features, item.num_targets
        m = int(item.season_length) if item.season_length else 1
        batch = eval_batch(item, params, p_out=p_out, l_pack=cfg.packing.L_pack).to(device)
        gen = torch.Generator(device=device).manual_seed(basis_seed)
        basis = make_basis(batch, cfg.model.d_model, device=device, generator=gen)
        out = forward(batch, variate_basis=basis)
        forecast = horizon_forecast_raw(out, batch, item, p_out=p_out).cpu()  # [p, nt]

        ch = nf  # first target channel
        ctx = item.data_tensor[ch].cpu()
        y_true = item.y_true[:, 0].cpu()
        snaive = seasonal_naive_forecast(item.data_tensor[ch], m, y_true.shape[0]).cpu()
        t_ctx, p = ctx.shape[0], y_true.shape[0]
        show = min(t_ctx, max(3 * m, 96))
        xs_ctx = range(t_ctx - show, t_ctx)
        xs_h = range(t_ctx, t_ctx + p)

        ax = axes[row][0]
        ax.plot(xs_ctx, ctx[-show:], color="0.6", lw=1, label="context")
        ax.plot(xs_h, y_true, color="black", lw=2, label="actual")
        ax.plot(xs_h, forecast[:, 0], color="tab:blue", marker=".", label="model")
        ax.plot(xs_h, snaive, color="tab:orange", ls="--", label="seasonal_naive")
        ax.axvline(t_ctx - 0.5, color="red", alpha=0.3, lw=1)
        ax.set_title(f"{item.config_id}   (C={nf + nt}, m={m})", fontsize=9)
        ax.legend(loc="upper left", fontsize=7, ncol=4)
    fig.suptitle(f"{cfg.run.name}: actual vs model vs seasonal-naive "
                 f"({len(idxs)} random eval samples)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
