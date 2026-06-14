"""Distributed training — DDP/FSDP, (node,rank) sharding, checkpoints (O6, S13).

Distributed is first-class, not a bolt-on: the seams have been honored since S6 —
the collator/model read no cross-rank or global state, packing/normalization/masks
are per-buffer and rank-local, and ``build_loader``/``StreamingReservoir`` already
shard series disjointly by ``(rank, world_size)`` with a rank-offset shuffle seed.
This module wires the remaining pieces:

- **Process group** (``init_distributed``/``cleanup_distributed``): gloo on
  CPU/Mac, nccl on CUDA.
- **Sharding** (``rank_shard``): the deterministic disjoint series partition (the
  same round-robin ``StandInPretrainLoader`` uses) — re-deriving it at a *new*
  world size re-shards with no overlap and full coverage.
- **Parallel wrap** (``wrap_model``): **DDP is the v1 default** (compile composes
  after the wrap). **FSDP is a recognized config switch** (``cfg.distributed.
  parallel``) wrapped only when selected *and* available — not exercised on CPU CI
  (FSDP-on-CPU is fragile), the same lazy posture as the GIFT-Eval download.
- **Cross-rank cost scheduling (D9.4 §9)** needs **no collective**: each rank's
  ``StreamingReservoir`` already cost-sorts its scheduler window (S11), so global
  step ``t`` draws similar-cost steps on every rank by construction — the
  "deterministic global cost-sorted schedule seeded identically" option.
- **Checkpoints (D13)**: ``save_checkpoint``/``load_checkpoint`` persist model +
  optimizer + **per-rank reservoir state**. Restoring at the *same* world size
  resumes the reservoir exactly; restoring at a *different* world size re-shards
  deterministically and the reservoir refills from the new shard.

Because nothing reads cross-rank state, a sample's forward/loss is **per-sample
numerically identical** whether run solo or as part of a multi-rank job — DDP only
averages gradients (``test_distributed``).
"""

from __future__ import annotations

import datetime
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from ..config import Config
from ..packing.reservoir import StreamingReservoir


def rank_shard(n_series: int, rank: int, world_size: int) -> List[int]:
    """Deterministic disjoint series indices for ``rank`` (round-robin, O6).

    Re-deriving this at a new ``world_size`` re-partitions with no double-counting
    and full coverage — the basis for checkpoint re-shard."""
    return list(range(rank, n_series, world_size))


def init_distributed(
    *,
    rank: int,
    world_size: int,
    backend: str = "gloo",
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    timeout_s: int = 120,
) -> Tuple[int, int]:
    """Initialize (or join) the process group; returns ``(rank, world_size)``."""
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_s),
        )
    return dist.get_rank(), dist.get_world_size()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The underlying module behind a DDP/FSDP wrap (``.module``), else ``model``."""
    return model.module if hasattr(model, "module") else model


def wrap_model(model: torch.nn.Module, cfg: Config, *, device: str = "cpu") -> torch.nn.Module:
    """Wrap ``model`` for the selected parallelism (DDP default; FSDP switch-only).

    DDP uses ``find_unused_parameters=True`` so steps whose buffer contains no
    tokens of some tier (that tier's encoder/head sees no grad) don't trip the
    reducer. ``torch.compile`` is applied by the caller *after* this wrap (CUDA)."""
    parallel = cfg.distributed.parallel
    if parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        if device.startswith("cuda"):
            return DDP(model, device_ids=[torch.cuda.current_device()],
                       find_unused_parameters=True)
        return DDP(model, find_unused_parameters=True)  # CPU/gloo
    if parallel == "fsdp":  # pragma: no cover - not exercised on CPU CI
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError as e:
            raise ImportError("cfg.distributed.parallel='fsdp' but FSDP is "
                              "unavailable in this torch build") from e
        return FSDP(model)
    raise ValueError(f"unknown cfg.distributed.parallel={parallel!r} (ddp | fsdp)")


# --- checkpoints (D13) --------------------------------------------------------

def _rank_path(path: str, rank: int) -> str:
    return f"{path}.rank{rank}.pt"


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    reservoir: Optional[StreamingReservoir],
    rank: int,
    world_size: int,
    step: int,
) -> str:
    """Persist model + optimizer + **per-rank reservoir state** (D13).

    Written one file per rank (model/optimizer are identical across DDP ranks; the
    reservoir state is rank-local). Returns the written path."""
    state = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "reservoir": reservoir.state_dict() if reservoir is not None else None,
        "world_size": world_size,
        "rank": rank,
        "step": step,
    }
    out = _rank_path(path, rank)
    torch.save(state, out)
    return out


def load_checkpoint(
    path: str,
    cfg: Config,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[StreamingReservoir, int]:
    """Restore model + optimizer; rebuild the reservoir for the current world size.

    Model/optimizer come from the canonical rank-0 file (identical across ranks).
    If the checkpoint's ``world_size`` matches the current one, the rank-local
    reservoir state is restored **exactly**; otherwise we **re-shard** — a fresh
    reservoir is built for the new ``(rank, world_size)`` and refills from its new
    disjoint shard (D13 mandatory different-world-size restart). Returns
    ``(reservoir, step)``."""
    base = torch.load(_rank_path(path, 0), weights_only=False)
    unwrap(model).load_state_dict(base["model"])
    if optimizer is not None and base.get("optimizer") is not None:
        optimizer.load_state_dict(base["optimizer"])

    reservoir = StreamingReservoir.from_cfg(cfg, rank=rank, world_size=world_size)
    same_world = int(base["world_size"]) == int(world_size)
    rank_file = _rank_path(path, rank)
    if same_world and os.path.exists(rank_file):
        rstate = torch.load(rank_file, weights_only=False).get("reservoir")
        if rstate is not None:
            reservoir.load_state_dict(rstate)  # exact resume
    # else: re-shard — reservoir is fresh for the new world size and refills.
    return reservoir, int(base["step"])
