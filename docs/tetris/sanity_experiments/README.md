# TETRIS — Sanity Experiments (architecture bring-up)

A running log proving the TETRIS architecture can actually **learn** before we
attempt the zero-shot protocol. Each experiment trains on a small pool of simple
**periodic synthetic** series and forecasts their held-out horizon, scored against
the **seasonal-naive** baseline with **MASE** (GIFT-Eval / gluonts style). The
season length `m` is *dataset metadata* (never detected by the model), exactly as
the GIFT-Eval test split provides it.

**Reading the metric:** `MASE = MAE_horizon / in-sample seasonal-naive denom` in
raw value space. `skill = model_MASE / snaive_MASE`; **skill < 1 ⇒ beats seasonal
naive**. We expect the model to *at least match* naive on clean periodic signals
and to *beat* it where there's learnable structure under the noise.

## Summary

| Case | Model (d/L/H, params) | Device | Steps | Time | model MASE | snaive MASE | skill | Verdict |
|---|---|---|---|---|---|---|---|---|
| [sine_univariate](#sine_univariate) | 64 / 3 / 4, 2.06M | CPU (4 thr) | 1500 | 1m47s | **0.812** | 0.981 | **0.83** | ✅ beats naive |
| multivariate_independent | — | — | — | — | — | — | — | pending |
| shared_factor | — | — | — | — | — | — | — | pending |
| features_target (+KFF) | — | — | — | — | — | — | — | pending |
| all cases mixed | — | — | — | — | — | — | — | pending |

## How to reproduce

Env: `uv` + Python 3.13 (see `docs/tetris/workflow.md`). Each run writes a
self-contained, git-ignored `outputs/<run>_<timestamp>/` with `command.txt`,
resolved `config.yaml`, `train_log.txt`, and `samples.png`.

```bash
uv run pytest                                              # expect all green
uv run python -m tetris.train.sanity_run configs/<case>.yaml --steps <N> --eval-every <K>
```

Each case below is a one-line config (`data.case` + `season_lengths`/`n_channels`),
CPU-friendly and identical model size unless noted. Plots referenced here are
copied from the run's `outputs/.../samples.png` (5 random eval samples: context +
actual vs model vs seasonal-naive).

---

## sine_univariate

One clean noisy sine per series, period `m=24` (noise scales with amplitude, so a
model that learns the underlying sine **beats** seasonal-naive, which propagates
the noise from one season ago). The easiest possible signal — the first proof that
the pipeline learns anything at all.

```bash
uv run python -m tetris.train.sanity_run configs/sanity_sine.yaml --steps 1500 --eval-every 500
```

Model: `d_model=64, n_layers=3, n_heads=4, out_patch=8` (2,057,188 params, all
trainable). Device: CPU, 4 threads, torch 2.12.0. 64 series, horizon 32.

| step | train_loss | model MASE | skill |
|---|---|---|---|
| 0 (random) | — | 6.795 | 6.93 |
| 500 | 0.325 | 0.919 | 0.94 |
| 1000 | 0.284 | 0.870 | 0.89 |
| 1500 | 0.249 | **0.812** | **0.83** |

MASE falls from 6.8 (random init) to 0.81, dropping below the seasonal-naive
baseline (0.98) — the model learns the periodic structure and extrapolates it
through the held-out horizon. 1500 steps in 1m47s (~72 ms/step).

![sine_univariate](plots/sine_univariate.png)
