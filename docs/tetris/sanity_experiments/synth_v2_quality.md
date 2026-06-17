# Synth-v2 Tier-1 quality report (H1, Mac CPU)

Tier-1 = a **training-free** verdict on whether the synthetic families cover the real
GIFT-Eval **test** data distribution, on the shared feature battery (`data/features.py`),
plus a **learnability** gate. Two principles drive it:

1. **Don't game the metric** ([[dont-game-synth-quality-metric]]): the headline **C2ST**
   (classifier two-sample test; AUC≈0.5 = indistinguishable) is on the **dynamics** feature
   subset (excludes `log_length`/`log_scale`), so a family can't look good by matching
   superficial series length/amplitude. A stronger **kNN** C2ST (the honest, higher number)
   and the full-feature AUC are reported alongside; a pure-noise control anchors the scale.
2. **Good synth must be _learnable_** — predictable structure + bounded noise, not random.
   The **learnability** gate scores forecastability by simple baselines (seasonal-naive /
   last-value / linear, median MASE on the tail horizon). Synth should be ~as forecastable as
   real, not wildly harder (random noise) or trivially easy.

Tier-2 (synth-only → real-test transfer probe) is the eventual falsifiable bar, deferred until
the synth is close.

## Result (seed 0, 500 series/family, real test via `$GIFT_EVAL`)

| family | C2ST dyn ↓ | C2ST dyn kNN ↓ | C2ST full | learnability (≈real) | RBF-MMD(dyn) |
|---|---|---|---|---|---|
| **targeted** | 0.935 | 0.991 | **0.941** | **1.196** | 0.091 |
| general | 0.933 | 0.973 | 0.970 | 1.169 | 0.052 |
| noise (control) | 0.998 | 0.999 | 0.998 | 0.870 | 0.042 |
| _real test_ | — | — | — | _1.169_ | — |

**Read:** the per-config rebuild made targeted **genuinely learnable** — its learnability
(1.20) matches real (1.17), i.e. its structure is forecastable by simple baselines like the real
data, and far from random noise. Targeted's **full** C2ST (0.941) beats general (0.970); the
**dynamics** C2ST is tied (~0.935), and the stricter kNN (0.991) shows fine structure is still
separable. So: structurally faithful + learnable, **not yet indistinguishable** — the remaining
gap is the fine autocorrelation/spectral shape (`acf1`/`acf_diff1`/`spectral_entropy` KS), the
H2 lever.

## The per-config targeted generator (rebuild)

Each config's profile now stores **aggregate** structural signatures (no values, no leakage):
**dominant periods** (top-K `(period, weight)` from the averaged periodogram) and **fitted AR(p)**
coefficients, alongside the feature quantiles + season/horizon/C marginals. `gen_targeted` then
**selects among three predictable-by-construction generators by goodness-of-fit**, gated by config
character:

- **spectral / dominant-component** — sum sinusoids (+harmonics) at the fitted dominant periods +
  bounded AR noise → regular multi-harmonic periodicity (LOOP_SEATTLE, M_DENSE, electricity cycles).
- **structural + AR** — level + trend (+ occasional level-shift) + seasonal + AR(p) noise from the
  fitted coefficients → trends / level-shifts / autocorrelation (electricity, covid_deaths, bizitobs).
- **regular spike-train** — flat AR baseline + spikes at the dominant period(s) with **fixed period,
  fixed phase, near-constant amplitude, minimal jitter** → *predictable* spikes (bitbrains server
  traces). Eligible **only for genuinely impulsive configs** (`excess_kurtosis > 0.5`), so it can't
  be mis-assigned to smooth/seasonal configs.

This fixed the two failures the visual + punch-list exposed: (a) earlier spikes were *random*
(jittered position/amplitude) → unlearnable noise; now they're *regular* → forecastable. (b) the
crude `kurt>0.4 & seas<0.3` impulsive trigger mis-fired on seasonal configs (SZ_TAXI); the kurtosis
gate + goodness-of-fit selection assigns the right regime per config.

## Open gaps → H2 levers (capture more real *dynamics*, never length/scale)

- **Fine autocorrelation / spectral shape** (`acf1`/`acf_diff1`/`spectral_entropy`): targeted is
  still slightly rougher than real near-unit-root/trending series (the AR-root shrink for stability
  costs smoothness). Lever: per-config parameter *fitting* (search generator knobs to minimize the
  feature/C2ST distance per group) and/or matched-spectrum noise.
- **Period × sampling-frequency decoupling** ([[synth-period-vs-sampling-frequency]]): a pattern's
  real-world period and the observation frequency jointly set samples-per-cycle (a daily cycle =
  period 24/96/288/1440 at H/15T/5T/1T). Generating across the **cross-product** of {temporal
  periodicity} × {sampling frequency} is a principled route to varied, realistic data.

## Visual sanity (like-for-like, 20 configs, same group per row)

`synth_targeted_vs_real.png` — 20 targeted (left) vs 20 real (right) over distinct configs, context
(blue) + horizon (orange-shaded), windowed to span ≥2 seasonal cycles. Targeted now reproduces
trends, regular seasonality, and **regular/predictable** spike trains where real is impulsive;
smooth/seasonal configs are no longer polluted by spurious spikes.

## Reproduction (Mac CPU)

```bash
# 1. fit the per-config TestProfile (season-aware feats + dominant periods + AR; aggregate, no values)
uv run python -m tetris.data.test_profile \
    --out src/tetris/data/profiles/gifteval_test.json --items-per-config 40
# 2. Tier-1 harness (dynamics C2ST + kNN + learnability + per-feature KS + noise control)
uv run python -m tetris.data.quality_harness \
    --profile src/tetris/data/profiles/gifteval_test.json --n 500 --items-per-config 30 \
    --report docs/tetris/sanity_experiments/synth_v2_quality.md
```

The profile JSON is **gitignored** (rebuild locally / rsync to WSL); it holds only aggregate
per-config statistics (feature quantiles, dominant periods, AR coefficients, marginals) — never raw
test values (no leakage).
