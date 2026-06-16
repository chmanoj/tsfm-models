# Synth-v2 Tier-1 quality report (H1, Mac CPU)

Tier-1 = a **training-free** verdict on whether the synthetic families cover the real
GIFT-Eval **test** data distribution, on the shared feature battery (`data/features.py`).
The headline metric is the **C2ST** (classifier two-sample test): a classifier is trained
to tell synth from real test on the feature vectors; **AUC ≈ 0.5 = indistinguishable**,
≈ 1.0 = trivially separable. Per-feature KS is the punch-list of *what still gives a family
away*; a pure-noise control anchors the scale.

**We do not game the metric** (see the [[dont-game-synth-quality-metric]] principle): the
headline AUC is computed on the **dynamics** feature subset (excludes `log_length`/`log_scale`),
so a family can't look good by matching superficial series length/amplitude that teach the
model no test *pattern*. We also report a stronger **kNN** C2ST (nonlinear, usually higher →
the honest number) and the full-feature AUC as context. Tier-2 (synth-only → real-test transfer
probe) is the eventual falsifiable bar, deferred until the synth is close.

## Result (seed 0, 500 series/family, real test via `$GIFT_EVAL`)

**Verdict:** `PASS(relative)` — `targeted dyn=0.914 / knn=0.974 | general dyn=0.931 / knn=0.971
| noise dyn=0.998`. Targeted is the **closest-to-test** family on the non-gameable dynamics
C2ST; the noise control is the most separable (the C2ST has discriminative power); but the best
dynamics-AUC ≈ 0.91 (kNN ≈ 0.97) is **still separable** — more real *dynamics* to capture.

| family | C2ST dyn ↓ | C2ST dyn kNN ↓ | C2ST full | RBF-MMD(dyn) |
|---|---|---|---|---|
| **targeted** | **0.914** | 0.974 | 0.931 | 0.073 |
| general | 0.931 | 0.971 | 0.968 | 0.053 |
| noise (control) | 0.998 | 0.999 | 0.998 | 0.041 |

### Per-feature KS (the punch-list — lower = closer to test)

| feature | general | targeted | noise |
|---|---|---|---|
| acf1 | **0.23** | 0.35 | 0.93 |
| acf_diff1 | **0.47** | 0.54 | 0.93 |
| spectral_entropy | **0.10** | 0.28 | 0.94 |
| dominant_period_frac | 0.21 | 0.20 | 0.60 |
| trend_strength | 0.26 | **0.07** | 0.51 |
| seasonal_strength | 0.09 | 0.14 | 0.47 |
| intermittency | 0.24 | 0.27 | 0.66 |
| excess_kurtosis | 0.69 | **0.53** | 0.89 |
| stationarity | 0.25 | 0.37 | 0.94 |
| _log_scale / log_length_ | _(0.50 / 0.72)_ | _(0.22 / 0.68)_ | — |

(`log_scale`/`log_length` shown in italics — **excluded** from the headline AUC on purpose.)

## What the harness-driven iteration found (collaborative debugging log)

Each fix below came from reading the punch-list and the like-for-like plot, *capturing real
dynamics* — not chasing the metric:

1. **Calendar-period seasonality (`features.season`).** Real server/traffic traces
   (e.g. `bitbrains_rnd/5T`) are dominated by a slow drift, so the FFT-*dominant* period is the
   trend and the daily (period-288) structure was invisible to a period-agnostic
   `seasonal_strength`. The feature now also measures strength at the **known calendar period**
   (and the profile fit passes it through), so the profile/generator can see real seasonality.
2. **Multi-harmonic / sharp seasonal shape.** A single sinusoid can't make real sharp daily
   peaks; the targeted seasonal component is now a 2–4 harmonic Fourier series, or a **von Mises
   peak train** for strongly-seasonal groups → matches the spectral/acf shape better.
3. **Impulsive / spike-train mode.** `bitbrains_rnd/5T` is a **flat baseline + periodic sharp
   spikes + rare huge outliers** (high `excess_kurtosis`, low seasonal variance), which smooth
   synthesis never produces. Added an impulsive generator regime (triggered by high kurtosis +
   low seasonal strength); the targeted samples now reproduce the periodic-spike character (see
   `synth_targeted_vs_real.png`, bottom rows). This dropped `trend_strength` KS to 0.07 and
   improved `excess_kurtosis`.
4. **Diversity mix.** ~40% of targeted draws are profile-rescaled samples from the diverse
   general families (joint-manifold coverage) — a single rigid family is trivially separable even
   when its marginals match (it pushed targeted C2ST from 0.99 → ~0.93).

### Open dynamics gaps (the real H2 data-quality levers — not length/scale)

- **`acf1` / `acf_diff1` / `spectral_entropy` / `stationarity`:** targeted is still rougher than
  the real series on fine autocorrelation/spectral shape. The smooth-regime synthesis needs
  better short-lag structure (e.g. fitted AR(p) / matched spectra) — candidate for **per-config
  parameter fitting** (search generative knobs per group to match the feature distribution).
- The kNN C2ST (~0.97) shows nonlinear separability remains; the linear dynamics-AUC (~0.91)
  is the gentler view. Both say: closer than general, not yet close.

## Visual sanity (like-for-like, same config group per row)

`synth_targeted_vs_real.png` — 10 targeted (left) vs 10 real (right), same config group per row,
context (blue) + horizon (orange-shaded), windowed to span ≥2 seasonal cycles. Targeted now
reproduces scale, trend, rough seasonality, and the impulsive spike character; fine
autocorrelation/spectral detail is the remaining gap.

## Reproduction (Mac CPU)

```bash
# 1. fit the per-config TestProfile (season-aware features; aggregate stats only, no values)
uv run python -m tetris.data.test_profile \
    --out src/tetris/data/profiles/gifteval_test.json --items-per-config 40
# 2. Tier-1 harness (dynamics C2ST + kNN + per-feature KS + noise control + predictability floor)
uv run python -m tetris.data.quality_harness \
    --profile src/tetris/data/profiles/gifteval_test.json --n 500 --items-per-config 30 \
    --report docs/tetris/sanity_experiments/synth_v2_quality.md
```

The profile JSON is **gitignored** (rebuild locally / rsync to WSL); it is aggregate per-config
feature quantiles + marginals, never raw test values (no leakage).
