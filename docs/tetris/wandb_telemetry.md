# TETRIS — wandb telemetry reference (G1)

What every logged panel means, how it's computed, and how to read them *together* to
debug a run. Added after the `gifteval_fixed_mix` 30M run "looked flat" but was
actually learning — the single-batch loss was just noisy/unsmoothed and the
leaderboard geomean was dominated by a handful of extrapolation blowups.

Code: per-step diagnostics in `train/diagnostics.py` + `train/loop.py` (`_step_scalars`);
per-eval in `train/tracking.py` (`eval_tail_scalars`, `per_config_rows`) + `data/eval_loader.py`
(the `fc_ctx_ratio`). Logged every `run.log_every` steps (auto = `min(eval_every//5, 50)`).

## The one mental model

The model predicts in **anchored-arcsinh space**: `z = arcsinh((x − a)/σ)` (anchor
`a` = median of recent context, `σ` = robust step-scale). Inversion is
`x = sinh(z)·σ + a`. **Training loss is in z-space; eval MASE is in real space.**
Because `sinh` is exponential, a small z-error at large `|z|` is a *huge* real-space
error the loss barely feels. Most debugging is "where in that gap is the problem."

## Per-step panels

| panel | how it's computed | what it means |
|---|---|---|
| `train_loss` | `horizon_MAE + Σ_k w_k·aux_MAE_k`, z-space, this batch | single-batch error; noisy by design — don't read trend off it |
| `train_loss_ema` | geometric EMA `expm1(0.98·ema_log + 0.02·log1p(loss))` | the de-noised trend — **this is "is it learning."** Geometric so the ~1e17 early transient can't poison it |
| `lr`, `steps_per_sec` | — | LR (const 1e-3) and throughput; throughput drop ⇒ heavier batch |
| `loss/horizon` | forecast head z-space MAE, masked by `target_valid` | the part you actually score; if this plateaus while aux falls, it's learning texture not forecasts |
| `loss/aux_total` | `Σ_k w_k·aux_k`, weights `[.2,.2,.2,.2,.1,.1]` | weighted next-patch auxiliary loss |
| `loss/aux_tier_0..5` | unweighted next-patch MAE per patch-tier (0=patch4 fine … 5=patch512 coarse) | **which timescale is stuck/unstable**; tiers are logged separately for exactly this |
| `grad_norm` | pre-clip global L2 of all grads (what `clip_grad_norm_` returns) | optimization stability; constantly ≫`grad_clip` ⇒ throttled every step; spikes ⇒ violent batch |
| `z/pred_p50,p99,absmax` | quantiles/max of `\|z_pred\|` over valid targets | model extremity in normalized space (healthy ≈ 0–7) |
| `z/tgt_absmax` | max `\|z_target\|` | how extreme the *data* is — the reference for `z/pred_*` |
| `pred/real_absmax` | max `\|sinh(z_pred)·σ + a\|` | largest real value emitted — blowup magnitude in actual units |
| `loss/real_mae` | `mean\|real_pred − real_tgt\|` over valid | **the train-side twin of MASE.** z-space loss ↓ but this stuck ⇒ the loss space hides the blowup |
| `sigma/p50,p99` | quantiles of `σ` over query tokens | scale-estimate extremity; high σ + modest z ⇒ blowup is a **scale** problem, not the model |
| `batch/n_var,n_tokens,L` | variate count / non-pad tokens / pack length | what was *in* the step; a spike on an odd-composition step is a batch artifact |

## Per-eval panels

| panel | how it's computed | what it means |
|---|---|---|
| `eval/leaderboard_mase` | geomean **across configs** of per-config `MASE = mean\|y−ŷ\| / mean_t\|y_t−y_{t−m}\|` | the headline; hides everything below |
| `eval/snaive_mase`, `eval/skill` | same for seasonal-naive; `skill = model/snaive` | <1 beats naive, >1 loses |
| `eval/n_configs[_in_gmean,_finite]` | counts | `finite` dropping ⇒ model emitted NaN/inf somewhere |
| `eval/mase_median` | median per-config MASE | the *typical* config (was 2.2 while geomean=38 ⇒ tail problem) |
| `eval/mase_p90, mase_max` | p90 / worst config MASE | the size of the tail |
| `eval/n_skill_gt1, n_skill_gt10` | # configs worse than naive / blowing up (>10×) | *counts* the problem instead of averaging it |
| `eval/fc_ctx_ratio_max, _median` | per config `max(forecast−a)/max(context−a)`, then max/median across configs | **the blowup attribution.** ≈1 = in historical range (inaccuracy); ≫1 = predicting values never seen (**extrapolation**) |
| `eval/per_config` (table) | `config_id, model_mase, snaive_mase, skill, fc_ctx_ratio, n_*`, worst-skill first | names the offenders + their fingerprint in one row |
| `eval/samples` (image) | forecast plots for N configs | eyeball *how* it fails (level shift, trend, seasonality) |

## Reading them together (playbooks)

1. **Is it learning?** `train_loss_ema` ↓ and `eval/leaderboard_mase` trajectory. Ignore raw `train_loss` jitter.
2. **Train ↓ but eval bad?** Compare `loss/horizon` (z) vs `loss/real_mae` (real). Both ↓ = distribution gap. z ↓, real stuck = **the loss space hides the blowup** (an argument about the loss, not the synth).
3. **Which configs, and why?** Sort `eval/per_config` by skill; read `fc_ctx_ratio`: ≫1 → extrapolation (synth fix: bounded-behaviour archetypes for that regime); ≈1 + tiny `snaive_mase` → worse-than-trivial on a near-flat series (different fix).
4. **Model or scale?** On a blowup, `z/pred_absmax` high + `sigma/p99` normal ⇒ model extrapolating; z normal + σ high ⇒ scale estimation wrong on short/low-freq context.
5. **Instability spikes?** Correlate `grad_norm` spike with `z/pred_absmax`, `loss/real_mae`, `batch/*` and the per-tier `loss/aux_tier_*` on the same step.
6. **Is the tail moving?** Watch `eval/mase_median` vs `mase_max`/`fc_ctx_ratio_max` separately — the geomean is tail-dominated, so only the tail panels tell you whether the score will move.

## Worked example — fine-tier early instability (10k run, 2026-06-21)

The per-tier panels caught something `train_loss` could never show. Around step
250–350 the **fine** next-patch tiers spiked while the coarse ones stayed calm:

```
step   t0      t1       t2       t3    t4    t5
 200   1.19    1.44     1.30     1.60  1.87  1.52
 250   742.29  1514.49  1915.18  2.05  2.23  2.41
 350   29.99   34.70    37.99    3.90  3.66  3.87
1000   1.05    1.22     1.11     1.28  1.57  1.44   (recovered)
```

The fine-patch heads (4/8/16) briefly explode while the model tames the global
transient (same window as the `grad_norm`/`z_absmax` spikes), then recover. Coarse
heads never do. Lead: fine-patch heads may want lr warmup or a head-specific clip.

**Note on viewing:** on a linear y-axis that ~1900 spike auto-scales the panel so the
fine-tier lines look pinned at zero everywhere else (reads as "no data for 0/1/2").
Use **log-y** for the `loss/aux_tier_*` panels.
