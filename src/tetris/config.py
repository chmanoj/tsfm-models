"""Configuration schema (dataclasses) + YAML load/merge via OmegaConf.

Plain dataclasses carry the heavy ablation-toggle surface (D10 norm axes, D13
mixture weights, the per-tier aux-weight vector, the distributed block). Every
combination must be reachable by a one-line YAML change with no code edits
(D10 mandate). ``load_config`` merges a structured schema with a user YAML file
so unspecified keys fall back to the full-size ``base`` defaults.

Note: OmegaConf structured configs require concrete typing (``typing.List`` /
``typing.Dict``), so this module deliberately does not use
``from __future__ import annotations``.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from omegaconf import OmegaConf

from .constants import PATCH, V


@dataclass
class RunCfg:
    name: str = "base"
    steps: int = 100000
    seed: int = 0


@dataclass
class BackendCfg:
    # auto -> cuda:flex+compile ; mps/cpu -> sdpa+eager (D14 + session convention)
    device: str = "auto"
    compile: bool = True  # honored on cuda; no-op on mps/cpu


@dataclass
class DistributedCfg:
    enabled: bool = False
    parallel: str = "ddp"          # ddp (v1 default) | fsdp (scale-up switch)
    process_group: str = "gloo"    # gloo on cpu/mac ; nccl on cuda
    shard_by: List[str] = field(default_factory=lambda: ["node", "rank"])


@dataclass
class ModelCfg:
    d_model: int = 256             # base; smokes use ~32 (O5)
    n_layers: int = 8              # base; smokes use 2
    n_heads: int = 4
    patch_vocab: List[int] = field(default_factory=lambda: list(PATCH))
    out_patch: int = 16            # P_out (D12 design point)
    # D12 allocation *ratio prior* (NOT literal counts/caps): per-tier token
    # counts are dynamic per segment = ratio * n, rebalanced to T_raw. This
    # vector normalizes to the design-point proportions ([8,8,8,8,8,4] at n=44).
    tier_alloc_per_channel: List[int] = field(default_factory=lambda: [16, 16, 16, 16, 16, 8])
    # Static per-tier encoder dispatch capacity (walkthrough Stage 7 ENCODER_CAP).
    # Each encoder_k runs [CAP, P_k, 2]→[CAP, D] at this fixed row count,
    # sentinel-padded + masked. 0 → resolve to packing.L_pack (the default = L).
    encoder_cap: int = 0


@dataclass
class DataCfg:
    loader: str = "standin_pretrain"
    n_series: int = 4096                # stand-in corpus size (real Pretrain is effectively infinite)
    C_distribution: List[int] = field(default_factory=lambda: [1, 21])
    length_distribution: List[int] = field(default_factory=lambda: [64, 100000])
    nan_cap: float = 0.3
    synthetic_mix: Dict[str, float] = field(
        default_factory=lambda: {"shared_factor": 0.5, "univariate": 0.4, "lag_probe": 0.1}
    )
    # D13 mixture: multiplier 0 removes a dataset (mass renormalized over survivors).
    dataset_weights: Dict[str, float] = field(default_factory=dict)
    # --- sanity stage (simple-synthetic train->test, scored vs seasonal naive) ---
    # The sanity loaders ('sanity' train / 'sanity_eval' eval) generate periodic
    # series whose season length is dataset metadata (never detected by the model),
    # mirroring GIFT-Eval's test split. `case` selects the generator family;
    # `season_lengths` is the calendar-style period pool (e.g. weekly=7, daily=24);
    # `horizon` is the held-out forecast length; `series_len` the per-series length.
    case: str = "sine_univariate"
    season_lengths: List[int] = field(default_factory=lambda: [24])
    horizon: int = 32
    series_len: int = 512
    n_channels: int = 4                  # C for multivariate sanity cases
    # Download root for the real GIFT-Eval tree (lazy; threaded to the eval loader).
    local_dir: str = ""


@dataclass
class PackingCfg:
    L_pack: int = 1024             # -> 2048 expansion (D12)
    buffers_per_step: int = 8      # B
    reservoir: bool = True         # trivial path flips this off (Stage 10)
    # Streaming packer (S11/D9.3): reservoir size K (doubles as shuffle buffer);
    # a buffer is emitted once its residual drops below tail_tolerance·L_pack
    # (or nothing else fits). Scheduler window W (64–256 buffers) is cost-sorted
    # (D9.4) and chunked into similar-cost B-buffer steps.
    reservoir_k: int = 1000        # K ≈ 1000 (D9.3)
    scheduler_window: int = 128    # W ∈ [64, 256] (D9.4)
    tail_tolerance: float = 0.05   # emit when residual < tail_tolerance·L_pack


@dataclass
class NormCfg:
    # D10: two independent axes + loss-space toggle. All combinations must run.
    input_norm: str = "anchored_arcsinh"     # | zscore_arcsinh
    loss_target: str = "locally_reanchored"  # | global_norm_space
    loss_space: str = "arcsinh"              # | vol_units
    anchor_window: int = 32                  # median-of-last-N anchor (8–16 tuning knob)


@dataclass
class LossCfg:
    # Per-tier aux weights (replaces a single lambda; D6 + raw-time aux note).
    aux_weights: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    loss_weighting: str = "none"  # D15 deferred: | seasonal_naive


@dataclass
class EvalCfg:
    enabled: bool = True
    loader: str = "gifteval_test"
    shard_windows: int = 100  # "first 100 windows per config" (record-only)


@dataclass
class ChecksCfg:
    assert_no_recompile: bool = True
    assert_pack_invariance: bool = True
    assert_aux_boundary: bool = True


@dataclass
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    backend: BackendCfg = field(default_factory=BackendCfg)
    distributed: DistributedCfg = field(default_factory=DistributedCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    packing: PackingCfg = field(default_factory=PackingCfg)
    norm: NormCfg = field(default_factory=NormCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    checks: ChecksCfg = field(default_factory=ChecksCfg)

    def __post_init__(self) -> None:
        if list(self.model.patch_vocab) != list(PATCH):
            raise ValueError(
                f"model.patch_vocab must equal the fixed D12 vocabulary {list(PATCH)}; "
                f"got {list(self.model.patch_vocab)}"
            )
        if len(self.model.tier_alloc_per_channel) != V:
            raise ValueError(
                f"model.tier_alloc_per_channel must have {V} entries; got {self.model.tier_alloc_per_channel}"
            )
        if any(w <= 0 for w in self.model.tier_alloc_per_channel):
            raise ValueError("model.tier_alloc_per_channel entries must be positive (ratio prior)")
        if self.model.encoder_cap < 0:
            raise ValueError("model.encoder_cap must be >= 0 (0 resolves to packing.L_pack)")
        if self.packing.reservoir_k < 1:
            raise ValueError("packing.reservoir_k must be >= 1")
        if self.packing.scheduler_window < 1:
            raise ValueError("packing.scheduler_window must be >= 1")
        if not (0.0 <= self.packing.tail_tolerance < 1.0):
            raise ValueError("packing.tail_tolerance must be in [0, 1)")
        if len(self.loss.aux_weights) != V:
            raise ValueError(f"loss.aux_weights must have {V} entries; got {self.loss.aux_weights}")
        if self.norm.input_norm not in ("anchored_arcsinh", "zscore_arcsinh"):
            raise ValueError(f"unknown norm.input_norm={self.norm.input_norm!r}")
        if self.norm.loss_target not in ("locally_reanchored", "global_norm_space"):
            raise ValueError(f"unknown norm.loss_target={self.norm.loss_target!r}")
        if self.norm.loss_space not in ("arcsinh", "vol_units"):
            raise ValueError(f"unknown norm.loss_space={self.norm.loss_space!r}")


def resolved_encoder_cap(cfg: "Config") -> int:
    """Static per-tier encoder dispatch capacity (ENCODER_CAP). 0 → L_pack."""
    return cfg.model.encoder_cap if cfg.model.encoder_cap > 0 else cfg.packing.L_pack


def load_config(path: str) -> Config:
    """Load a YAML file and merge it onto the structured ``base`` defaults,
    returning a validated ``Config`` dataclass instance."""
    schema = OmegaConf.structured(Config)
    user = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, user)
    # to_object instantiates the dataclasses, triggering __post_init__ validation.
    return OmegaConf.to_object(merged)
