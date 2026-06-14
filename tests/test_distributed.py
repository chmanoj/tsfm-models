"""S13 test_distributed — real spawned-gloo DDP, sharding, checkpoint re-shard.

Spawns real gloo process groups (``torch.multiprocessing``) on CPU and asserts the
O6 guarantees: source series shard disjointly and cover the corpus per rank; a
sample's forward/loss is **per-sample numerically identical** under DDP vs a single
process (no cross-rank/global state); and a checkpoint saves per-rank reservoir
state and **re-shards at a different world size** (2 → 3 ranks) with model/optimizer
restored. Real-gloo throughout (maintainer choice); FSDP is switch-only and not
exercised here.
"""

import os
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from tetris.config import load_config
from tetris.data.standin_loader import StandInPretrainLoader
from tetris.model.tetris import Tetris
from tetris.packing.reservoir import StreamingReservoir, packed_batches
from tetris.train import distributed as D
from tetris.train.shakedown import next_batch, sampler_params
from tetris.train.step import make_basis, mark_dynamic_batch, train_step

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
N_SERIES = 60


def _cfg():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.packing.L_pack = 256
    cfg.packing.buffers_per_step = 2
    cfg.packing.reservoir_k = 8
    cfg.packing.scheduler_window = 4
    cfg.model.out_patch = 8
    cfg.model.d_model = 16
    cfg.model.n_layers = 1
    cfg.model.n_heads = 2
    cfg.data.n_series = N_SERIES
    cfg.data.length_distribution = [64, 128]
    cfg.data.C_distribution = [1, 3]
    return cfg


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run(fn, world_size, *args):
    """Spawn ``world_size`` gloo workers; re-raises any worker exception.

    ``spawn`` calls ``fn(rank, *args)``; we thread ``world_size`` in as the second
    positional so workers receive ``(rank, world_size, ...)``."""
    mp.spawn(fn, args=(world_size, *args), nprocs=world_size, join=True)


def _fixed_batch(cfg):
    """A deterministic unsharded batch — identical in every process (rank-blind)."""
    loader = StandInPretrainLoader.from_cfg(cfg, rank=0, world_size=1)
    return next_batch(iter(loader), cfg, sampler_params(cfg), np.random.default_rng(0))


# --- workers (module-level so torch.multiprocessing.spawn can pickle them) -----

def _w_shards(rank, world_size, port, outdir):
    D.init_distributed(rank=rank, world_size=world_size, master_port=port)
    shard = D.rank_shard(N_SERIES, rank, world_size)
    # The loader's own shard length must match the deterministic partition.
    loader = StandInPretrainLoader(n_series=N_SERIES, rank=rank, world_size=world_size)
    assert len(loader) == len(shard)
    torch.save(shard, os.path.join(outdir, f"shard_{rank}.pt"))
    D.cleanup_distributed()


def _w_forward_loss(rank, world_size, port, outdir):
    D.init_distributed(rank=rank, world_size=world_size, master_port=port)
    cfg = _cfg()
    torch.manual_seed(0)
    model = Tetris(cfg)
    ddp = D.wrap_model(model, cfg, device="cpu")
    ddp.eval()
    batch = _fixed_batch(cfg)
    gen = torch.Generator().manual_seed(1)
    basis = make_basis(batch, cfg.model.d_model, generator=gen)
    with torch.no_grad():
        out = ddp(batch, variate_basis=basis)
    from tetris.losses import compute_loss

    loss = float(compute_loss(out, batch, aux_weights=cfg.loss.aux_weights).total)
    if rank == 0:
        torch.save(loss, os.path.join(outdir, "ddp_loss.pt"))
    D.cleanup_distributed()


def _train_a_few(cfg, ddp, opt, reservoir, steps):
    batches = packed_batches(reservoir, l_pack=cfg.packing.L_pack,
                             p_out=cfg.model.out_patch,
                             num_buffers=cfg.packing.buffers_per_step)
    done = 0
    for batch in batches:
        if done >= steps:
            break
        gen = torch.Generator().manual_seed(1)
        basis = make_basis(batch, cfg.model.d_model, generator=gen)
        mark_dynamic_batch(batch, basis)
        train_step(ddp, batch, basis, opt, aux_weights=cfg.loss.aux_weights)
        done += 1


def _w_ckpt_save(rank, world_size, port, ckpt_path, outdir):
    D.init_distributed(rank=rank, world_size=world_size, master_port=port)
    cfg = _cfg()
    torch.manual_seed(0)
    ddp = D.wrap_model(Tetris(cfg), cfg, device="cpu")
    opt = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    res = StreamingReservoir.from_cfg(cfg, rank=rank, world_size=world_size)
    _train_a_few(cfg, ddp, opt, res, steps=2)
    D.save_checkpoint(ckpt_path, model=ddp, optimizer=opt, reservoir=res,
                      rank=rank, world_size=world_size, step=2)
    if rank == 0:
        sig = float(next(D.unwrap(ddp).parameters()).detach().sum())
        torch.save(sig, os.path.join(outdir, "sig_saved.pt"))
    D.cleanup_distributed()


def _w_ckpt_load(rank, world_size, port, ckpt_path, outdir):
    D.init_distributed(rank=rank, world_size=world_size, master_port=port)
    cfg = _cfg()
    ddp = D.wrap_model(Tetris(cfg), cfg, device="cpu")
    opt = torch.optim.Adam(ddp.parameters(), lr=1e-3)
    res, step = D.load_checkpoint(ckpt_path, cfg, model=ddp, optimizer=opt,
                                  rank=rank, world_size=world_size)
    assert step == 2
    if rank == 0:  # capture the restored weights *before* any further training
        sig = float(next(D.unwrap(ddp).parameters()).detach().sum())
        torch.save(sig, os.path.join(outdir, "sig_loaded.pt"))
    _train_a_few(cfg, ddp, opt, res, steps=1)  # reservoir refills on the new shard
    torch.save(D.rank_shard(N_SERIES, rank, world_size),
               os.path.join(outdir, f"reshard_{rank}.pt"))
    D.cleanup_distributed()


# --- tests --------------------------------------------------------------------

def test_disjoint_shards_cover_corpus(tmp_path):
    _run(_w_shards, 2, _free_port(), str(tmp_path))
    s0 = torch.load(tmp_path / "shard_0.pt")
    s1 = torch.load(tmp_path / "shard_1.pt")
    assert set(s0).isdisjoint(s1)                     # no series double-counted
    assert sorted(s0 + s1) == list(range(N_SERIES))   # full coverage


def test_ddp_forward_loss_matches_single_process(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    ref_model = Tetris(cfg)
    ref_model.eval()
    batch = _fixed_batch(cfg)
    basis = make_basis(batch, cfg.model.d_model, generator=torch.Generator().manual_seed(1))
    from tetris.losses import compute_loss

    with torch.no_grad():
        ref = float(compute_loss(ref_model(batch, variate_basis=basis), batch,
                                 aux_weights=cfg.loss.aux_weights).total)

    _run(_w_forward_loss, 2, _free_port(), str(tmp_path))
    ddp_loss = torch.load(tmp_path / "ddp_loss.pt")
    assert abs(ddp_loss - ref) < 1e-5                 # per-sample loss rank-independent


def test_checkpoint_restore_reshard_2_to_3(tmp_path):
    ckpt = str(tmp_path / "ckpt")
    _run(_w_ckpt_save, 2, _free_port(), ckpt, str(tmp_path))    # train @ ws=2, save
    _run(_w_ckpt_load, 3, _free_port(), ckpt, str(tmp_path))    # restore @ ws=3

    sig_saved = torch.load(tmp_path / "sig_saved.pt")
    sig_loaded = torch.load(tmp_path / "sig_loaded.pt")
    assert abs(sig_saved - sig_loaded) < 1e-6          # model restored across world sizes

    shards = [torch.load(tmp_path / f"reshard_{r}.pt") for r in range(3)]
    flat = sorted(i for s in shards for i in s)
    assert flat == list(range(N_SERIES))               # re-shard: disjoint + full coverage
    for r in range(3):
        for q in range(r + 1, 3):
            assert set(shards[r]).isdisjoint(shards[q])
