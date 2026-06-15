"""GIFT-Eval ``test``-split OVERFIT entrypoint (G3 — in-distribution capacity probe).

Ties the test-as-training loader (``data.loader: gifteval_test_overfit``) to the real
GIFT-Eval test windows (``eval.loader: gifteval_test``) and reports the **leaderboard
MASE** — the per-config MASE geo-meaned across configs (G2), vs the seasonal-naive
baseline. This is *not* zero-shot: the model trains on the same series' contexts it is
scored on. The point is to prove we can learn real data at all and watch the
random-init ``inf`` leaderboard MASE become finite (and log the skill so we know where
we stand). Honest framing lives in the G3 reconciliation block.

    # smoke (WSL): 500 steps, eval every 250
    uv run python -m tetris.train.overfit_run configs/gifteval_test_overfit.yaml \
        --steps 500 --eval-every 250
    # full run (WSL): 20k steps, eval every 5k
    uv run python -m tetris.train.overfit_run configs/gifteval_test_overfit.yaml \
        --steps 20000 --eval-every 5000

Each run writes a git-ignored ``outputs/<name>_<ts>/`` with the command, resolved
config, full log, the final ``model.pt`` checkpoint, and the sample-forecast plot
(also uploaded to wandb alongside the scalars). CPU/eager by default; flip
``backend.compile`` + a CUDA device for the GPU run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch
from omegaconf import OmegaConf

from ..config import load_config
from ..data.contract import build_eval_loader, build_loader
from ..data.eval_loader import evaluate_leaderboard
from ..model.tetris import Tetris
from .loop import run_training
from .sanity_plot import plot_eval_samples
from .sanity_run import (
    _count_params, _device_info, _fmt_duration, _make_run_dir, resolve_device,
    setup_logging,
)
from .tracking import eval_scalars, make_tracker

log = logging.getLogger("tetris.overfit")


def _fmt(r: dict) -> str:
    return (f"leaderboard_MASE={r['leaderboard_mase']:.4f}  snaive_MASE={r['snaive_mase']:.4f}  "
            f"skill={r['skill']:.4f}  (configs={r['n_configs_finite']}/{r['n_configs_in_gmean']}"
            f"/{r['n_configs']} finite/in-gmean/scored, skipped={r['skipped']})")


def _log_per_config(r: dict, *, top: int = 0) -> None:
    """Dump the per-config MASE breakdown to the train log (sorted by skill)."""
    pc = r.get("per_config", {})
    if not pc:
        return
    rows = sorted(pc.items(), key=lambda kv: (kv[1]["skill"] != kv[1]["skill"], kv[1]["skill"]))
    if top:
        rows = rows[:top]
    log.info("per-config leaderboard (%d configs):", len(pc))
    for cid, d in rows:
        log.info("  %-40s model=%.4f snaive=%.4f skill=%.4f (n=%d, ch_items=%d)",
                 cid, d["model_mase"], d["snaive_mase"], d["skill"],
                 d["n_items"], d["n_channel_items"])


def run_overfit(cfg_path: str, *, steps: int = 0, lr: float = 1e-3,
                eval_every: int = 0, device: str = None, n_plot: int = 5,
                compile: bool = None) -> dict:
    cfg = load_config(cfg_path)
    steps = steps or cfg.run.steps
    eval_every = eval_every or max(1, steps // 4)
    device = device or resolve_device(cfg.backend.device)
    if compile is not None:
        cfg.backend.compile = bool(compile)

    run_dir = _make_run_dir(cfg)
    (run_dir / "command.txt").write_text(
        "uv run python -m tetris.train.overfit_run " + " ".join(sys.argv[1:]) + "\n"
        f"# resolved: steps={steps} lr={lr} eval_every={eval_every} device={device}\n"
    )
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    setup_logging(run_dir / "train_log.txt")

    torch.manual_seed(cfg.run.seed)
    model = Tetris(cfg).to(device)
    # Real GIFT-Eval test windows (leaderboard) — lazy/network (needs $GIFT_EVAL + extras).
    eval_loader = build_eval_loader(cfg, local_dir=cfg.data.local_dir or None)
    total, trainable = _count_params(model)

    tracker = make_tracker(
        cfg, run_name=run_dir.name, run_dir=run_dir,
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
    )

    # Real CUDA path (D14): FlexAttention + torch.compile of the *training* forward.
    # Eval/plot stay eager (B=1); evaluate_leaderboard marks R/n_var dynamic + hoists
    # the block mask so the leaderboard logs under --compile too (G1 fix; G2 _score_item).
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
    log.info("data: loader=%s terms=%s items_per_config=%d | eval_items=%d | steps=%d lr=%g eval_every=%d",
             cfg.data.loader, list(cfg.data.terms), cfg.eval.items_per_config,
             len(eval_loader), steps, lr, eval_every)
    try:
        train_loader = build_loader(cfg)
        log.info("train loader: %d test-window contexts (cyclic overfit)", len(train_loader))
    except Exception as e:  # don't let a loader-size probe abort the run
        log.info("train loader: size unknown (%s)", e)

    base = evaluate_leaderboard(model, eval_loader, cfg, device=device)
    log.info("step %5d (random init): %s", 0, _fmt(base))
    tracker.log_scalars(eval_scalars(base), step=0)

    log_every = max(1, eval_every // 5)

    def _on_log(step, loss):
        log.info("step %5d/%d: train_loss=%.4f", step, steps, loss)

    def _on_eval(step, r, loss):
        log.info("step %5d/%d: train_loss=%.4f  %s  [eval]", step, steps, loss, _fmt(r))

    t0 = time.perf_counter()
    losses = run_training(
        cfg, steps=steps, lr=lr, device=device, model=model, forward=forward_train,
        eval_loader=eval_loader, eval_every=eval_every, tracker=tracker,
        eval_fn=evaluate_leaderboard, log_every=log_every, on_log=_on_log, on_eval=_on_eval,
    )
    elapsed = time.perf_counter() - t0
    n_done = len(losses)
    rate = n_done / elapsed if elapsed > 0 else float("nan")
    log.info("training complete: %d steps in %s (%.2f steps/s, %.1f ms/step)",
             n_done, _fmt_duration(elapsed), rate, 1000.0 * elapsed / max(1, n_done))

    final = evaluate_leaderboard(model, eval_loader, cfg, device=device)
    log.info("FINAL: %s  => %s seasonal naive", _fmt(final),
             "BEATS" if final["skill"] < 1 else "does NOT beat")
    _log_per_config(final)
    tracker.log_scalars(eval_scalars(final), step=steps)

    # Final checkpoint (entrypoint-level; D13 per-rank reservoir state is single-rank
    # here so a plain model state_dict suffices — distributed.save_checkpoint is G5).
    ckpt = run_dir / "model.pt"
    torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(
        OmegaConf.structured(cfg), resolve=True), "step": n_done}, ckpt)
    log.info("saved checkpoint -> %s", ckpt)

    plot_path = plot_eval_samples(
        model, eval_loader, cfg, out_path=str(run_dir / "samples.png"),
        n=n_plot, seed=cfg.run.seed, device=device,
    )
    log.info("wrote plots -> %s", plot_path)
    tracker.log_image("eval/samples", plot_path, step=steps)  # uploaded alongside scalars
    tracker.finish()
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="TETRIS GIFT-Eval test-split overfit (leaderboard MASE)")
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-plot", type=int, default=5)
    ap.add_argument("--compile", dest="compile", action="store_true", default=None,
                    help="force torch.compile on (CUDA Flex path); default reads backend.compile")
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="force torch.compile off")
    args = ap.parse_args()
    run_overfit(args.config, steps=args.steps, lr=args.lr, eval_every=args.eval_every,
                device=args.device, n_plot=args.n_plot, compile=args.compile)


if __name__ == "__main__":
    main()
