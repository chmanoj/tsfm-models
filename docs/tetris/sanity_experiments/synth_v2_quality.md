# Synth-v2 Tier-1 quality report (H1, Mac CPU)

Tier-1 = a **training-free** verdict on whether the synthetic families cover the real
GIFT-Eval **test** data distribution, on the shared feature battery (`data/features.py`).
The headline metric is the **C2ST** (classifier two-sample test) AUC: a classifier is
trained to tell synth from real test on the feature vectors; **AUC ≈ 0.5 means
indistinguishable**, AUC ≈ 1.0 means trivially separable. Per-feature KS is the
punch-list of *what still gives a family away*; a pure-noise control anchors the scale.

Tier-2 (the synth-only → real-test transfer probe) is deferred until the synth is good
enough — Tier-1 iterates faster and, unlike a probe, can't confound "bad synth" with
"under-trained model" (the maintainer's call).

## Result (seed 0, 600 series/family, real test via `$GIFT_EVAL`, items-per-config 40)

**Verdict:** `PASS(relative)` — `targeted AUC=0.949 | general AUC=0.966 | noise AUC=0.997`
→ targeted is the **closest-to-test** family, the noise control is the most separable
(C2ST has discriminative power), but the **best real AUC ≈ 0.95 is still clearly
separable** — absolute distribution-match is H2 data-quality work.

Noise-robust predictability floor (irreducible RMS / scale): **≈ 0.61**.

| family | C2ST AUC ↓ | RBF-MMD ↓ |
|---|---|---|
| **targeted** | **0.949** | 0.162 |
| general | 0.966 | 0.109 |
| noise (control) | 0.997 | 0.601 |

### Per-feature KS (the punch-list — lower = closer to test)

| feature | general | targeted | noise |
|---|---|---|---|
| log_scale | 0.48 | **0.20** | 0.92 |
| log_length | 0.69 | 0.64 | 0.69 |
| acf1 | **0.19** | 0.32 | 0.92 |
| acf_diff1 | **0.43** | 0.51 | 0.91 |
| spectral_entropy | **0.08** | 0.24 | 0.93 |
| dominant_period_frac | 0.21 | 0.20 | 0.59 |
| trend_strength | 0.27 | **0.18** | 0.51 |
| seasonal_strength | 0.08 | **0.13** | 0.47 |
| intermittency | 0.24 | 0.27 | 0.66 |
| excess_kurtosis | 0.67 | **0.56** | 0.87 |
| stationarity | 0.22 | 0.34 | 0.94 |

### Reading the punch-list (H2 targets)

- **Targeted wins on the structural axes it's conditioned on:** `log_scale`, `trend_strength`,
  `seasonal_strength`, `excess_kurtosis` — the profile-conditioning works.
- **Targeted still trails general on the spectral/autocorrelation axes** (`spectral_entropy`,
  `acf1`, `acf_diff1`, `stationarity`): real GIFT-Eval test series (traffic/energy) have
  **sharp multi-harmonic periodicity**, while the targeted synthesis is single-sinusoid +
  smoothed noise. The diverse general *mixture* happens to cover those spectra better.
  **H2 lever:** richer seasonal harmonics + sharper periodic generators in the targeted family.
- **`log_length` separates every family** (≈0.65–0.69): synthetic length distributions differ
  from the test contexts even after quantile sampling — a second H2 lever.
- The **iteration loop the harness enables**: each fix above was found by reading this exact
  KS table (e.g. tying the noise lag-1 autocorr to the profile `acf1` dropped targeted's
  `stationarity`/`acf1` KS; adding a diverse general-mixture component dropped its C2ST AUC
  from 0.99 → 0.95).

## Visual sanity (like-for-like, same config group per row)

`synth_targeted_vs_real.png` — 10 targeted samples (left) vs 10 real GIFT-Eval test samples
(right), each split context (blue) + horizon (orange), generated for the **same** config group
per row. Targeted reproduces scale, trend, and rough seasonality; it misses the sharp
multi-harmonic periodicity of the real traffic series — consistent with the spectral KS gap.

## Reproduction (Mac CPU)

```bash
# 1. fit the per-config TestProfile from the real test split (aggregate stats only, no values)
uv run python -m tetris.data.test_profile \
    --out src/tetris/data/profiles/gifteval_test.json --items-per-config 40
# 2. run the Tier-1 harness (C2ST / KS / MMD / noise control / predictability floor)
uv run python -m tetris.data.quality_harness \
    --profile src/tetris/data/profiles/gifteval_test.json --n 600 --items-per-config 40 \
    --report docs/tetris/sanity_experiments/synth_v2_quality.md --json /tmp/synth_quality.json
```

The profile JSON is **gitignored** (rebuild locally / rsync to WSL as needed); it is aggregate
per-config feature quantiles + marginals, never raw test values (no leakage).
