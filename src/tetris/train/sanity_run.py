"""Sanity training entrypoint — simple-synthetic train->test, scored vs seasonal naive.

Ties the sanity train loader (``data.loader: sanity``) to the matched eval shard
(``eval.loader: sanity_eval``) and reports **MASE** of the model against the
seasonal-naive baseline (the season length is dataset metadata). The smallest
end-to-end proof that the architecture can learn: forecast the held-out horizon of
a small pool of periodic series and beat (or at least match) seasonal naive.

    uv run python -m tetris.train.sanity_run configs/sanity_sine.yaml --steps 400

Every run writes a self-contained, git-ignored directory under ``outputs/`` with
the exact command, the resolved config, the full training log, and the
actual-vs-model-vs-seasonal-naive plots for 5 random eval samples.

CPU/eager by default; flip ``backend.compile`` + a CUDA device for the GPU run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from omegaconf import OmegaConf

from ..config import Config, load_config
from ..data.contract import build_eval_loader, build_loader
from ..data.eval_loader import evaluate_mase
from ..model.tetris import Tetris
from .loop import run_training
from .sanity_plot import plot_eval_samples
from .tracking import eval_scalars, make_tracker

OUTPUTS_ROOT = Path("outputs")
log = logging.getLogger("tetris.sanity")

_LOG_FMT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_path: Path, *, level: int = logging.INFO) -> None:
    """Configure ``tetris.*`` logging: console + the run's ``train_log.txt``,
    formatted ``<time> | <level> | <module> | <msg>`` (standard training-log style)."""
    fmt = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)
    root = logging.getLogger("tetris")
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    fileh = logging.FileHandler(log_path, mode="w")
    fileh.setFormatter(fmt)
    root.setLevel(level)
    root.addHandler(stream)
    root.addHandler(fileh)
    root.propagate = False


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_log_every(cfg, eval_every: int) -> int:
    """Train-metric log cadence (G1): ``cfg.run.log_every`` if pinned (>0), else the
    auto default ``min(eval_every//5, 50)`` — frequent enough to see the curve, but
    never coarser than a fifth of the eval cadence on small runs."""
    pinned = int(getattr(cfg.run, "log_every", 0) or 0)
    return pinned if pinned > 0 else min(max(1, eval_every // 5), 50)


def _fmt(r: dict) -> str:
    return (f"model_MASE={r['model_mase']:.4f}  snaive_MASE={r['snaive_mase']:.4f}  "
            f"skill={r['skill']:.4f}  (n={r['n']})")


def _count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _device_info(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"{device} ({name}, {gb:.1f} GiB, torch {torch.__version__})"
    return f"{device} ({torch.get_num_threads()} cpu threads, torch {torch.__version__})"


def _fmt_duration(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return (f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s")


def _log_loader_sizes(train_loader, eval_loader, cfg) -> None:
    """Best-effort data-loader sizes (optional — loaders need not expose ``__len__``)."""
    try:
        pool = len(train_loader)
        cyclic = getattr(train_loader, "cycle", False)
        ctx_len = cfg.data.series_len - cfg.data.horizon
        log.info("train loader: pool=%d series%s, context_len=%d (~%s ctx steps/epoch)",
                 pool, " (cyclic)" if cyclic else "", ctx_len, f"{pool * ctx_len:,}")
    except (TypeError, AttributeError):
        log.info("train loader: size unknown (streaming)")
    try:
        log.info("eval loader: %d items, horizon=%d", len(eval_loader), cfg.data.horizon)
    except (TypeError, AttributeError):
        pass


def _make_run_dir(cfg: Config) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUTS_ROOT / f"{cfg.run.name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_sanity(cfg_path: str, *, steps: int = 0, lr: float = 1e-3,
               eval_every: int = 0, device: str = None, n_plot: int = 3,
               compile: bool = None) -> dict:
    cfg = load_config(cfg_path)
    steps = steps or cfg.run.steps
    eval_every = eval_every or max(1, steps // 4)
    device = device or resolve_device(cfg.backend.device)
    if compile is not None:
        cfg.backend.compile = bool(compile)

    run_dir = _make_run_dir(cfg)
    # Reproducibility artifacts: exact command + resolved config.
    (run_dir / "command.txt").write_text(
        "uv run python -m tetris.train.sanity_run " + " ".join(sys.argv[1:]) + "\n"
        f"# resolved: steps={steps} lr={lr} eval_every={eval_every} device={device}\n"
    )
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    setup_logging(run_dir / "train_log.txt")

    torch.manual_seed(cfg.run.seed)
    model = Tetris(cfg).to(device)
    eval_loader = build_eval_loader(cfg, local_dir=cfg.data.local_dir or None)
    total, trainable = _count_params(model)

    # Experiment tracker (G1): default wandb online->offline->disabled (no-op).
    # Logs the resolved config + scalars; never a hard dependency (CI/Mac safe).
    tracker = make_tracker(
        cfg, run_name=run_dir.name, run_dir=run_dir,
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
    )

    # Real CUDA path (D14): FlexAttention + torch.compile. Compile only the
    # *training* forward (one graph after warmup; R/n_var marked dynamic by the
    # step). Eval/plot stay eager (B=1, per-item shapes) — same params, used as the
    # correctness reference. Compile is a no-op off CUDA.
    compiled = bool(cfg.backend.compile) and device.startswith("cuda")
    forward_train = model
    if compiled:
        log.info("torch.compile: ON (Flex backend, dynamic R/n_var) — first steps include compile")
        forward_train = torch.compile(model)

    log.info("run_dir=%s", run_dir)
    log.info("device: %s", _device_info(device))
    log.info("model: d_model=%d n_layers=%d n_heads=%d out_patch=%d | params=%s (trainable=%s)",
             cfg.model.d_model, cfg.model.n_layers, cfg.model.n_heads, cfg.model.out_patch,
             f"{total:,}", f"{trainable:,}")
    log.info("data: case=%s m=%s n_series=%d horizon=%d series_len=%d | steps=%d lr=%g eval_every=%d",
             cfg.data.case, list(cfg.data.season_lengths), cfg.data.n_series, cfg.data.horizon,
             cfg.data.series_len, steps, lr, eval_every)
    _log_loader_sizes(build_loader(cfg), eval_loader, cfg)
    base = evaluate_mase(model, eval_loader, cfg, device=device)
    log.info("step %5d (random init): %s", 0, _fmt(base))
    tracker.log_scalars(eval_scalars(base), step=0)

    log_every = resolve_log_every(cfg, eval_every)

    def _on_log(step, loss):
        log.info("step %5d/%d: train_loss=%.4f", step, steps, loss)

    def _on_eval(step, r, loss):
        log.info("step %5d/%d: train_loss=%.4f  %s  [eval]", step, steps, loss, _fmt(r))

    # Mid-train eval uses the training forward, compiled or not: evaluate_mase marks
    # R/n_var dynamic and hoists the block mask (like the train step), so under
    # --compile the B=1 eval reuses one graph instead of recompiling per item — so
    # the periodic eval MASE now logs on the GPU too (G1 GPU-note #1 fix).
    t0 = time.perf_counter()
    losses = run_training(
        cfg, steps=steps, lr=lr, device=device, model=model, forward=forward_train,
        eval_loader=eval_loader, eval_every=eval_every, tracker=tracker,
        eval_fn=evaluate_mase, log_every=log_every, on_log=_on_log, on_eval=_on_eval,
    )
    elapsed = time.perf_counter() - t0
    n_done = len(losses)
    rate = n_done / elapsed if elapsed > 0 else float("nan")
    log.info("training complete: %d steps in %s (%.2f steps/s, %.1f ms/step)",
             n_done, _fmt_duration(elapsed), rate, 1000.0 * elapsed / max(1, n_done))

    final = evaluate_mase(model, eval_loader, cfg, device=device)
    log.info("FINAL: %s  => %s seasonal naive", _fmt(final),
             "BEATS" if final["skill"] < 1 else "does NOT beat")
    tracker.log_scalars(eval_scalars(final), step=steps)

    # KFF counterfactual: re-eval the SAME model with the feature future toggled,
    # to show known-future covariates are actually needed (D11). Only when the case
    # has feature channels.
    if eval_loader[0].num_features > 0:
        import copy

        cf = copy.deepcopy(cfg)
        cf.data.known_future_features = not cfg.data.known_future_features
        cf_loader = build_eval_loader(cf, local_dir=cf.data.local_dir or None)
        cf_res = evaluate_mase(model, cf_loader, cf, device=device)
        on, off = (final, cf_res) if cfg.data.known_future_features else (cf_res, final)
        log.info("KFF contrast: KFF-on model_MASE=%.4f  vs  past-only model_MASE=%.4f  "
                 "(KFF cuts error %.0f%%)",
                 on["model_mase"], off["model_mase"],
                 100.0 * (1 - on["model_mase"] / off["model_mase"]) if off["model_mase"] else float("nan"))

    plot_path = plot_eval_samples(
        model, eval_loader, cfg, out_path=str(run_dir / "samples.png"),
        n=n_plot, seed=cfg.run.seed, device=device,
    )
    log.info("wrote plots -> %s", plot_path)
    tracker.finish()
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="TETRIS sanity train->test (MASE vs seasonal naive)")
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-plot", type=int, default=3)
    ap.add_argument("--compile", dest="compile", action="store_true", default=None,
                    help="force torch.compile on (CUDA Flex path); default reads backend.compile")
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="force torch.compile off")
    args = ap.parse_args()
    run_sanity(args.config, steps=args.steps, lr=args.lr, eval_every=args.eval_every,
               device=args.device, n_plot=args.n_plot, compile=args.compile)


if __name__ == "__main__":
    main()
