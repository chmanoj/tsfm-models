# Tier-1 synth-quality report

Real GIFT-Eval test feature rows: **11906**

**Verdict:** PASS(relative) — targeted:dyn=0.932/knn=0.987 | general:dyn=0.935/knn=0.966 | noise:dyn=0.999/knn=0.999 || best targeted/general dynamics-AUC=0.932 → still separable — capture more real dynamics (not length/scale) (0.5 = indistinguishable; dyn excludes length/scale)

Noise-robust predictability floor (irreducible RMS / scale): **0.607**

## Learnability (best-baseline MASE on the tail horizon — lower = more forecastable; should be near real, not ≫)

real test: **1.269**

| family | learnability |
|---|---|
| noise | 0.876 |
| general | 1.157 |
| targeted | 1.180 |

## C2ST / MMD (lower AUC = closer to test; 0.5 = indistinguishable). Headline = **dynamics** AUC (length/scale excluded — not gameable).

| family | n | C2ST dyn | C2ST dyn (kNN) | C2ST full | RBF-MMD(dyn) |
|---|---|---|---|---|---|
| targeted | 793 | 0.932 | 0.987 | 0.938 | 0.0819 |
| general | 981 | 0.935 | 0.966 | 0.967 | 0.0510 |
| noise | 400 | 0.999 | 0.999 | 0.999 | 0.0334 |

## Per-feature KS distance (the punch-list: what still separates)

| feature | general | targeted | noise |
|---|---|---|---|
| log_scale | 0.51 | 0.19 | 0.89 |
| log_length | 0.79 | 0.68 | 0.79 |
| acf1 | 0.25 | 0.54 | 0.95 |
| acf_diff1 | 0.47 | 0.49 | 0.93 |
| spectral_entropy | 0.05 | 0.46 | 0.95 |
| dominant_period_frac | 0.22 | 0.25 | 0.56 |
| trend_strength | 0.28 | 0.14 | 0.49 |
| seasonal_strength | 0.33 | 0.17 | 0.66 |
| intermittency | 0.31 | 0.31 | 0.60 |
| excess_kurtosis | 0.70 | 0.56 | 0.91 |
| stationarity | 0.27 | 0.55 | 0.96 |
