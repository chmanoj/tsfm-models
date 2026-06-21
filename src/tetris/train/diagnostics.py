"""Per-step training diagnostics (G1 — Phase 1 cheap logging).

Turns one train step's forward output + batch into a flat dict of finite scalars
for the experiment tracker, so a flat-looking loss curve can be *explained* rather
than guessed at. All quantities are detached, computed under ``no_grad``, and
reduced to Python floats. Two families:

- **z-space vs real-space horizon error.** The training loss is MAE in anchored
  *arcsinh (z) space*; the leaderboard metric is real-space MASE. ``sinh`` is
  exponential, so a modest z-error at large ``|z|`` is a huge real-space error the
  loss barely sees. Logging both side-by-side is what tells you whether a blowup is
  even visible to the optimizer. ``z_pred_*`` flags model extrapolation; ``real_*``
  the magnitude it denormalizes to.
- **Scale + composition context** (σ quantiles, variate count, token fill) so loss
  spikes can be correlated with *what was in the batch*.

Grad-norm is captured in ``train_step`` (it needs the live grads), not here.
"""

from __future__ import annotations

from typing import Dict

import torch

from ..normalize import horizon_invert


def _q(v: torch.Tensor, qs) -> torch.Tensor:
    """Quantiles of a 1-D float tensor (empty -> NaNs). The ``q`` tensor must live on
    the input's device (CUDA quantile rejects a CPU ``q``)."""
    if v.numel() == 0:
        return torch.full((len(qs),), float("nan"))
    return torch.quantile(v.float(), torch.tensor(qs, dtype=torch.float32, device=v.device))


@torch.no_grad()
def horizon_diagnostics(output, batch) -> Dict[str, float]:
    """Cheap per-step horizon diagnostics over the valid query targets.

    Returns a flat dict (NaNs dropped by the caller's scalar filter). Keys:
      ``z/pred_p50|p99|absmax``  — |z_pred| over valid targets (extrapolation watch)
      ``z/tgt_absmax``           — |z_target| (how extreme the data itself is)
      ``pred/real_absmax``       — max |denormalized pred| (real-space blowup watch)
      ``loss/real_mae``          — real-space horizon MAE (the metric-space view)
      ``sigma/p50|p99``          — per-token base scale σΔ[0] over query tokens
      ``batch/n_var|n_tokens|L`` — composition (correlate spikes with batch content)
    """
    z = output.horizon.detach()                       # [B, L, P_out]
    z_tgt = batch.horizon_target                       # [B, L, P_out]
    valid = batch.target_valid                         # [B, L, P_out] bool
    a = batch.stats_a.unsqueeze(-1)                    # [B, L, 1]
    sig = batch.stats_sigma.unsqueeze(-1)             # [B, L, 1]

    zv = z[valid]
    ztv = z_tgt[valid]
    zq = _q(zv.abs(), [0.5, 0.99])
    out: Dict[str, float] = {
        "z/pred_p50": float(zq[0]),
        "z/pred_p99": float(zq[1]),
        "z/pred_absmax": float(zv.abs().max()) if zv.numel() else float("nan"),
        "z/tgt_absmax": float(ztv.abs().max()) if ztv.numel() else float("nan"),
    }

    real_pred = horizon_invert(z, a, sig)[valid]
    real_tgt = horizon_invert(z_tgt, a, sig)[valid]
    if real_pred.numel():
        out["pred/real_absmax"] = float(real_pred.abs().max())
        out["loss/real_mae"] = float((real_pred - real_tgt).abs().mean())

    # σ over the tokens that actually carry a query target.
    qtok = valid.any(dim=-1)                            # [B, L]
    sq = _q(batch.stats_sigma[qtok], [0.5, 0.99])
    out["sigma/p50"] = float(sq[0])
    out["sigma/p99"] = float(sq[1])

    out["batch/n_var"] = float(int(batch.variate_uid.max()) + 1)
    out["batch/n_tokens"] = float(int((batch.sample_id >= 0).sum()))
    out["batch/L"] = float(batch.sample_id.shape[1])
    return out


def grad_global_norm(params) -> torch.Tensor:
    """L2 norm of all grads (the no-clip path's stand-in for what
    ``clip_grad_norm_`` returns). Skips params without a grad."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.norm(torch.stack([g.detach().norm() for g in grads]))
