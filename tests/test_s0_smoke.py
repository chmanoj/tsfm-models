"""S0 smoke: config instantiates from shakedown.yaml; attend() runs on random
tensors on the active (Mac/SDPA or CUDA/Flex) backend."""

import os

import torch

from tetris.backend import attend, backend_kind, resolve_device
from tetris.config import load_config

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_config_loads_shakedown():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    assert cfg.run.name == "shakedown"
    assert cfg.model.d_model == 32
    assert cfg.model.patch_vocab == [4, 8, 16, 64, 256, 512]
    assert cfg.packing.L_pack == 256
    # D10 axes present and valid
    assert cfg.norm.input_norm == "anchored_arcsinh"
    assert len(cfg.loss.aux_weights) == 6


def test_config_loads_base():
    cfg = load_config(os.path.join(CONFIG_DIR, "base.yaml"))
    assert cfg.model.d_model == 256
    assert cfg.model.n_layers == 8
    assert cfg.packing.L_pack == 1024


def test_attend_runs():
    device = resolve_device("auto")
    kind = backend_kind(device)
    # SDPA path on Mac/CPU; this test exercises whatever backend is active.
    B, H, L, dh = 2, 2, 16, 8
    q = torch.randn(B, H, L, dh, device=device)
    k = torch.randn(B, H, L, dh, device=device)
    v = torch.randn(B, H, L, dh, device=device)
    out = attend(q, k, v, kind=kind)
    assert out.shape == (B, H, L, dh)
    assert torch.isfinite(out).all()
