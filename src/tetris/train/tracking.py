"""Experiment-tracker seam (G1) — log scalars + config from training.

A thin, backend-agnostic seam so the training loop can record loss / eval MASE /
lr / throughput plus the run config without hard-wiring a vendor. The default
backend is **wandb** with graceful degradation (the maintainer's G1 choice):

    online   -> network + login probe succeeds
    offline  -> probe fails (writes a local ``wandb/`` dir, no account; sync later)
    disabled -> ``wandb`` isn't importable at all (a logged no-op)

so CI + Mac dev never depend on the tracker. ``cfg.tracking.backend == "none"``
forces the no-op tracker. A tracker problem never crashes a run — it degrades to
:class:`NullTracker` and logs why (lands in the run's ``train_log.txt``).

Selected via :func:`make_tracker`; the loop logs through the :class:`Tracker`
protocol (``log_config`` / ``log_scalars`` / ``finish``).
"""

from __future__ import annotations

import logging
import math
import os
import socket
from typing import Optional, Protocol

log = logging.getLogger("tetris.tracking")


class Tracker(Protocol):
    """Minimal sink: the config once, scalars (+ images) per step, finish at the end."""

    def log_config(self, config: dict) -> None: ...
    def log_scalars(self, scalars: dict, *, step: int) -> None: ...
    def log_image(self, key: str, path, *, step: int) -> None: ...
    def finish(self) -> None: ...


class NullTracker:
    """No-op tracker (tracking disabled / CI / Mac). Every call is cheap."""

    def log_config(self, config: dict) -> None:
        pass

    def log_scalars(self, scalars: dict, *, step: int) -> None:
        pass

    def log_image(self, key: str, path, *, step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbTracker:
    """wandb-backed tracker (online or offline); thin wrapper over one run."""

    def __init__(self, wandb, run) -> None:
        self._wandb = wandb
        self._run = run

    def log_config(self, config: dict) -> None:
        self._run.config.update(config, allow_val_change=True)

    def log_scalars(self, scalars: dict, *, step: int) -> None:
        self._run.log(dict(scalars), step=step)

    def log_image(self, key: str, path, *, step: int) -> None:
        """Upload a saved image (e.g. the sample-forecast plot) under ``key``.
        Best-effort: a bad path/image never crashes a run."""
        try:
            self._run.log({key: self._wandb.Image(str(path))}, step=step)
        except Exception as e:  # pragma: no cover - wandb/image edge cases
            log.warning("tracker.log_image(%r) failed: %s", key, e)

    def finish(self) -> None:
        self._run.finish()


def make_tracker(cfg, *, run_name: str, run_dir, config: Optional[dict] = None) -> Tracker:
    """Build the tracker for this run from ``cfg.tracking`` (G1).

    Never raises on a tracker problem — degrades to :class:`NullTracker` and logs
    why, so logging can't take down a training run."""
    tcfg = getattr(cfg, "tracking", None)
    backend = getattr(tcfg, "backend", "none") if tcfg is not None else "none"
    if backend in ("none", "off", "disabled"):
        log.info("experiment tracking: disabled (backend=none)")
        return NullTracker()
    if backend != "wandb":
        log.warning("experiment tracking: disabled (unknown backend %r)", backend)
        return NullTracker()
    return _make_wandb(tcfg, run_name=run_name, run_dir=run_dir, config=config)


def _make_wandb(tcfg, *, run_name, run_dir, config) -> Tracker:
    try:
        import wandb
    except Exception:  # not installed / broken install -> disable, don't crash
        log.warning("experiment tracking: disabled (wandb not installed — "
                    "`uv pip install wandb` to enable)")
        return NullTracker()
    _load_dotenv()  # make WANDB_API_KEY in a local .env discoverable (login + online)
    mode = _resolve_mode(getattr(tcfg, "mode", "auto"))
    if mode == "disabled":
        log.info("experiment tracking: disabled (mode=disabled)")
        return NullTracker()
    try:
        run = wandb.init(
            project=getattr(tcfg, "project", "tetris"),
            name=run_name,
            dir=str(run_dir),
            mode=mode,
            config=config,
        )
    except Exception as e:  # network/login/etc — degrade rather than crash a run
        log.warning("experiment tracking: disabled (wandb.init failed: %s)", e)
        return NullTracker()
    log.info("experiment tracking: wandb mode=%s project=%s run=%s",
             mode, getattr(tcfg, "project", "tetris"), run_name)
    return WandbTracker(wandb, run)


def _resolve_mode(mode: str) -> str:
    """``auto`` -> ``online`` iff logged-in AND wandb host reachable, else
    ``offline``. Explicit online|offline|disabled pass through; ``WANDB_MODE`` wins."""
    env = os.environ.get("WANDB_MODE")
    if env:
        return env
    if mode != "auto":
        return mode
    if _logged_in() and _reachable("api.wandb.ai", 443):
        return "online"
    return "offline"


def _load_dotenv(path: str = ".env") -> None:
    """Best-effort load of ``KEY=VALUE`` pairs from a local ``.env`` into the
    environment **without overriding** already-set vars, so the wandb API key in
    ``.env`` is discoverable (login + online mode). Zero-dependency; silently does
    nothing if the file is absent/unreadable. Runs are launched from the repo root,
    so the default relative ``.env`` resolves there (mirror it onto the WSL box —
    rsync includes it — to track GPU runs online)."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def _logged_in() -> bool:
    """True if a wandb API key is discoverable (env or ``~/.netrc``)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc

        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except Exception:
        return False


def _reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """A quick TCP probe so ``auto`` falls to offline fast when the net is down."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def eval_scalars(result, *, prefix: str = "eval/") -> dict:
    """Flatten an eval result into finite numeric scalars for the tracker.

    Accepts an ``evaluate_mase`` dict (model/snaive MASE, skill, n, …) or a bare
    scalar loss; drops non-numeric and NaN values (e.g. an undefined skill)."""
    if isinstance(result, dict):
        return {f"{prefix}{k}": float(v) for k, v in result.items()
                if isinstance(v, (int, float)) and not math.isnan(float(v))}
    return {f"{prefix}test_loss": float(result)}
