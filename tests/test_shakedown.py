"""S10 test_shakedown — the compiled train step + tiny end-to-end runner.

Covers: streaming trivial-path shakedown runs end-to-end (finite losses); the
model can learn (overfits a fixed batch); and **no torch._dynamo recompiles** under
varying per-step shapes (D14 one-graph) — the basis is hoisted out of the compiled
forward and ``R``/``n_var`` are marked dynamic, so the frame count is flat after
warmup even as the raw-store width and variate count change.
"""

import os

import numpy as np
import torch
import torch._dynamo as dynamo
from torch._dynamo.testing import CompileCounter

from tetris.config import load_config
from tetris.data.contract import build_loader
from tetris.model.tetris import Tetris
from tetris.packing.collator import pack
from tetris.tokenize.assemble import assemble
from tetris.tokenize.window_sampler import SamplerParams, sample_window
from tetris.train.shakedown import next_batch, run_shakedown, sampler_params
from tetris.train.step import make_basis, mark_dynamic_batch, train_step

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
P_OUT = 8


def _cfg():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.packing.L_pack = 256
    cfg.packing.buffers_per_step = 2
    cfg.model.out_patch = P_OUT
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.model.n_heads = 2
    return cfg


def test_shakedown_runs_end_to_end():
    torch.manual_seed(0)
    losses = run_shakedown(_cfg(), steps=6, device="cpu")
    assert len(losses) == 6
    assert all(np.isfinite(losses))


def test_overfits_single_batch():
    cfg = _cfg()
    prior = tuple(cfg.model.tier_alloc_per_channel)

    def seg(C, T, s):
        g = torch.Generator().manual_seed(s)
        item = (torch.randn(C, T, generator=g).cumsum(1).to(torch.float32), 0, C)
        rng = np.random.default_rng(s)
        spec = sample_window(0, C, T, SamplerParams(l_pack=256, p_out=P_OUT, tier_prior=prior), rng)
        return assemble(item, spec, P_OUT)

    batch = pack([[seg(2, 400, 1)], [seg(1, 900, 2)]], l_pack=256, p_out=P_OUT)
    torch.manual_seed(0)
    model = Tetris(cfg)
    basis = make_basis(batch, cfg.model.d_model, generator=torch.Generator().manual_seed(1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = [
        float(train_step(model, batch, basis, opt, aux_weights=cfg.loss.aux_weights).total.detach())
        for _ in range(80)
    ]
    assert losses[-1] < 0.3 * losses[0]  # learns the fixed batch


def test_no_recompile_under_varying_shapes():
    cfg = _cfg()
    dynamo.reset()
    torch.manual_seed(0)
    model = Tetris(cfg)
    counter = CompileCounter()
    compiled = torch.compile(model, backend=counter, dynamic=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    params = sampler_params(cfg)
    rng = np.random.default_rng(0)
    it = iter(build_loader(cfg))

    def one_step():
        b = next_batch(it, cfg, params, rng)
        basis = make_basis(b, cfg.model.d_model, generator=torch.Generator().manual_seed(1))
        mark_dynamic_batch(b, basis)
        train_step(compiled, b, basis, opt, aux_weights=cfg.loss.aux_weights)
        return b.norm_values.shape[1], basis.shape[1]

    one_step(); one_step()                      # warmup: dynamo settles on a dynamic graph
    baseline = counter.frame_count
    assert baseline >= 1
    shapes = [one_step() for _ in range(4)]      # varying R and n_var
    assert counter.frame_count == baseline       # no recompiles (D14)
    assert len({r for r, _ in shapes}) > 1        # the test actually varied R
