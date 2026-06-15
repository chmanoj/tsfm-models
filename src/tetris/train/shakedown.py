"""Tiny end-to-end shakedown runner (S10 entrypoint).

The **trivial no-reservoir path** (S10): stream items from ``build_loader`` →
``sample_window`` → ``assemble`` → group **one segment per buffer** → ``pack`` →
train step. The reservoir + best-fit-decreasing path (S11) reuses the *same*
``pack`` and ``train_step``; only the grouping changes. Stays rank-local (O6).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from ..config import Config
from ..data.contract import build_loader
from ..model.tetris import Tetris
from ..packing.collator import Batch, pack
from ..tokenize.assemble import assemble
from ..tokenize.window_sampler import SamplerParams, sample_window
from .step import make_basis, mark_dynamic_batch, train_step


def sampler_params(cfg: Config) -> SamplerParams:
    return SamplerParams(
        l_pack=cfg.packing.L_pack,
        p_out=cfg.model.out_patch,
        tier_prior=tuple(cfg.model.tier_alloc_per_channel),
        max_query_tokens=cfg.packing.max_query_tokens,
    )


def next_batch(loader_iter, cfg: Config, params: SamplerParams, rng) -> Optional[Batch]:
    """Pull ``B`` items and pack them one-segment-per-buffer (trivial path)."""
    B = cfg.packing.buffers_per_step
    segs = []
    for item in loader_iter:
        data, nf, nt = item
        spec = sample_window(nf, nt, data.shape[1], params, rng)
        segs.append(assemble(item, spec, cfg.model.out_patch))
        if len(segs) == B:
            break
    if not segs:
        return None
    return pack([[s] for s in segs], l_pack=cfg.packing.L_pack, p_out=cfg.model.out_patch)


def run_shakedown(
    cfg: Config,
    *,
    steps: int,
    lr: float = 1e-3,
    device: str = "cpu",
    forward=None,
    model: Optional[Tetris] = None,
    generator: Optional[torch.Generator] = None,
) -> List[float]:
    """Run ``steps`` train steps on streaming synthetic; return per-step total loss.

    ``forward``/``model`` default to a fresh eager :class:`Tetris`; pass a compiled
    forward (and its model for the optimizer) to exercise the compiled path.
    """
    if model is None:
        model = Tetris(cfg).to(device)
    if forward is None:
        forward = model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    params = sampler_params(cfg)
    rng = np.random.default_rng(cfg.run.seed)
    loader_iter = iter(build_loader(cfg))

    losses: List[float] = []
    for _ in range(steps):
        batch = next_batch(loader_iter, cfg, params, rng)
        if batch is None:
            break
        batch = batch.to(device)
        basis = make_basis(batch, cfg.model.d_model, device=device, generator=generator)
        mark_dynamic_batch(batch, basis)
        lb = train_step(
            forward, batch, basis, optimizer,
            aux_weights=cfg.loss.aux_weights, loss_space=cfg.norm.loss_space,
        )
        losses.append(float(lb.total.detach()))
    return losses


def main() -> None:  # pragma: no cover - manual entrypoint
    import argparse
    from ..config import load_config

    ap = argparse.ArgumentParser(description="TETRIS shakedown")
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    losses = run_shakedown(cfg, steps=args.steps, device=args.device)
    print(f"steps={len(losses)} first={losses[0]:.4f} last={losses[-1]:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
