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
    """Minimal sink: the config once, scalars (+ images/tables) per step, finish."""

    def log_config(self, config: dict) -> None: ...
    def log_scalars(self, scalars: dict, *, step: int) -> None: ...
    def log_image(self, key: str, path, *, step: int) -> None: ...
    def log_table(self, key: str, columns, rows, *, step: int) -> None: ...
    def finish(self) -> None: ...


class NullTracker:
    """No-op tracker (tracking disabled / CI / Mac). Every call is cheap."""

    def log_config(self, config: dict) -> None:
        pass

    def log_scalars(self, scalars: dict, *, step: int) -> None:
        pass

    def log_image(self, key: str, path, *, step: int) -> None:
        pass

    def log_table(self, key: str, columns, rows, *, step: int) -> None:
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

    def log_table(self, key: str, columns, rows, *, step: int) -> None:
        """Log a sortable table (e.g. the per-config leaderboard breakdown).
        Best-effort: a table problem never crashes a run."""
        try:
            self._run.log({key: self._wandb.Table(columns=list(columns), data=[list(r) for r in rows])},
                          step=step)
        except Exception as e:  # pragma: no cover - wandb/table edge cases
            log.warning("tracker.log_table(%r) failed: %s", key, e)

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


# Columns of the per-config leaderboard table (G1 per-eval diagnostics).
PER_CONFIG_COLUMNS = ["config_id", "model_mase", "snaive_mase", "skill",
                      "fc_ctx_ratio", "n_items", "n_channel_items"]


def _finite(xs):
    return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]


def _pct(xs, q):
    """Plain percentile of a list (no numpy dep); empty -> NaN."""
    s = sorted(xs)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def eval_tail_scalars(result, *, prefix: str = "eval/") -> dict:
    """Tail/blowup summary from a leaderboard ``per_config`` dict (G1).

    The headline geo-mean hides which configs are pathological. This derives, across
    configs: the MASE spread (median/p90/max), how many configs are worse than
    seasonal-naive (skill>1) or blowing up (skill>10), and the extrapolation ratio
    (max/median forecast-vs-context magnitude) — the "why it exploded" signals. Pure
    post-processing of ``per_config``; ``{}`` if absent. NaNs dropped by the caller."""
    pc = result.get("per_config") if isinstance(result, dict) else None
    if not pc:
        return {}
    mases = _finite(d.get("model_mase") for d in pc.values())
    skills = _finite(d.get("skill") for d in pc.values())
    ratios = _finite(d.get("fc_ctx_ratio") for d in pc.values())
    out = {
        f"{prefix}mase_median": _pct(mases, 0.5),
        f"{prefix}mase_p90": _pct(mases, 0.9),
        f"{prefix}mase_max": max(mases) if mases else float("nan"),
        f"{prefix}n_skill_gt1": float(sum(1 for s in skills if s > 1)),
        f"{prefix}n_skill_gt10": float(sum(1 for s in skills if s > 10)),
        f"{prefix}fc_ctx_ratio_max": max(ratios) if ratios else float("nan"),
        f"{prefix}fc_ctx_ratio_median": _pct(ratios, 0.5),
    }
    return {k: v for k, v in out.items() if isinstance(v, (int, float)) and math.isfinite(v)}


def per_config_rows(result, *, top: int = 0):
    """Rows for the per-config leaderboard table, worst skill first (G1).

    Returns ``(columns, rows)`` aligned to :data:`PER_CONFIG_COLUMNS`; ``top`` caps
    the row count (0 = all). ``([], [])`` if the result carries no ``per_config``."""
    pc = result.get("per_config") if isinstance(result, dict) else None
    if not pc:
        return PER_CONFIG_COLUMNS, []
    def _skill(kv):
        s = kv[1].get("skill", float("nan"))
        nan = s != s
        return (nan, -s if not nan else 0.0)  # worst (highest) skill first, NaN last
    rows = []
    for cid, d in sorted(pc.items(), key=_skill):
        rows.append([cid, d.get("model_mase"), d.get("snaive_mase"), d.get("skill"),
                     d.get("fc_ctx_ratio"), d.get("n_items"), d.get("n_channel_items")])
    if top:
        rows = rows[:top]
    return PER_CONFIG_COLUMNS, rows
