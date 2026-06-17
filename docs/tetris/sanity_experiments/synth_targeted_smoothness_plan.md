# Targeted-synth smoothness — plan of action (H1.1 working doc)

**Status:** discussion captured 2026-06-16, *not yet implemented*. This doc is the
durable handoff for the H1.1 generation work. Read it together with
`prompts/synth_scale_H1_1.md`, the Synth-v2 H1 reconciliation block in
`docs/tetris/implementation_plan.md`, and the Tier-1 report
`docs/tetris/sanity_experiments/synth_v2_quality.md`. Principles:
[[dont-game-synth-quality-metric]], [[synth-period-vs-sampling-frequency]],
[[visual-first-synth-quality]].

Evidence committed alongside this doc:
- `synth_smoothness_diagnostic.png` — per-config real (blue) vs current targeted synth
  (orange), 8 configs across the smoothness spectrum (the picture that drove this plan).
- `synth_v2_quality_baseline_H1_1.md` — the baseline Tier-1 harness numbers (below).

---

## 1. The diagnosis (what we actually found)

The maintainer's call: **don't trust the stats alone — look at the data plots**, or we'll
fall into a stat-hacking loop (the [[dont-game-synth-quality-metric]] trap). We did, and the
picture changed the problem statement.

**Visual finding (the headline).** Real GIFT-Eval series are **smooth / high-SNR**: smooth
slow drifts, clean Gaussian-like bumps, smooth monotone growth, smooth daily/weekly waves, or
**quiet baselines with rare sharp spikes**. Our current targeted synth **drowns every config
in a pervasive high-frequency noise floor** — it always wiggles. Per config:

| config (season m) | REAL looks like | CURRENT SYNTH looks like |
|---|---|---|
| jena_weather/10T (144) | smooth slow wander, ~no HF noise | jagged noise |
| electricity/H (24) | near-flat baseline + a few sharp spikes | continuous noise oscillation |
| bizitobs_service (360) | clean Gaussian-like bumps | rough noise |
| LOOP_SEATTLE/5T (288) | smooth daily wave + gentle trend | seasonal but rough/spiky |
| bitbrains_rnd/H (24) | quiet baseline + rare isolated spikes | constant noise, no quiet baseline |
| covid_deaths (1) | ultra-smooth monotone growth | noisy upward drift |
| m4_daily | smooth trend + hump | trend but bumpy texture |

**Stats finding (confirms, but understates).** From the baseline harness KS punch-list, on the
*smoothness* features the targeted family is **worse than the generic family**:

| feature | general KS | **targeted KS** |
|---|---|---|
| acf1 | 0.25 | **0.54** |
| spectral_entropy | 0.05 | **0.46** |
| stationarity | 0.27 | **0.55** |

(targeted *wins* on seasonal_strength 0.17, trend 0.14, log_scale 0.19 — the structure it was
built for — but loses on smoothness.) Baseline dynamics-C2ST: targeted 0.932 / general 0.935;
kNN 0.987 / 0.966. Learnability: targeted 1.18, general 1.16, **real 1.27** (we are slightly
*easier* than real, consistent with too-regular structure + wrong noise).

**The reframe.** The gap is **not** "match the acf1 number" or "add more periods." It is:
1. our additive noise is **too large** and **the wrong spectral color** (white / AR(1) jitter
   vs. real residual variation which is *smooth / low-frequency / near-integrated*);
2. we never produce a **genuinely quiet baseline** (impulsive configs need silence between
   spikes; our spike baseline `0.25×AR` is too loud);
3. we have **no smooth-deterministic-envelope** generator (logistic growth, Gaussian bumps).

The sampling-domain lens explains *why* this is concentrated where it is
([[synth-period-vs-sampling-frequency]]): smoothness (acf1↑, spectral_entropy↓, stationarity↓)
is governed by **samples-per-cycle**. A correlation length fixed in *wall-clock time* becomes a
high acf1 automatically at fine sampling (5T/10T/15T), and a near-flat in-window drift when the
real-world cycle is longer than the window. Generating in **continuous time then sampling at the
config's interval** makes smoothness fall out by construction instead of being hand-matched.

> **Anti-stat-hack guard (carry into every change):** if a knob moves a KS/C2ST number but the
> side-by-side plot does **not** look more like real, reject it. acf1≈0.99 is satisfied equally
> by a smooth sine, a slow drift, and a sawtooth — the number is necessary, never sufficient.

---

## 2. Dataset clusters (work by groups of similar dynamics, not by frequency)

Cluster by *dynamics regime* (this is more useful than freq, and ties to the smoothness axis).
Sorted per-config smoothness stats (q50) are in `synth_v2_quality_baseline_H1_1.md`'s source
profile; the regimes:

- **A — ultra-smooth / near-unit-root (fine-sampled or smooth growth).** jena_weather/10T,
  bizitobs_l2c/5T, solar/10T, ett2/15T, ett2/H, bizitobs_application, bizitobs_service,
  m4_daily, covid_deaths. acf1 0.98–0.998, stationarity ~0.004–0.03. Shapes: smooth wander,
  smooth monotone growth, clean bumps.
- **B — smooth seasonal (clear cycle, low noise).** electricity/15T, LOOP_SEATTLE/5T, solar/H,
  kdd_cup/H, M_DENSE/H, m4_hourly, m4_monthly, ett1/15T, ett1/H, jena_weather/H. acf1 0.82–0.96.
  Shapes: smooth daily/weekly waves, often + slow trend.
- **C — trend / level-shift dominated (coarse, aperiodic).** electricity/D, electricity/W,
  m4_weekly/quarterly/yearly, saugeenday/D/W, us_births/W. Smooth ramps, level shifts.
- **D — impulsive (quiet baseline + rare sharp spikes).** bitbrains_rnd/5T+H,
  bitbrains_fast_storage/5T+H, electricity/H (spiky), temperature_rain. Near-silent baseline,
  isolated spikes — **regular** where periodic, sparse-random where not.
- **E — noisy / intermittent / count.** restaurant, car_parts, hospital, hierarchical_sales/D+W,
  solar/D (night zeros), us_births/D, SZ_TAXI. Low acf1, high stationarity, intermittent counts,
  day/night zeroing.

Work **one cluster at a time**: inspect 4–8 real exemplars, decide the generative recipe that
reproduces *that* regime, regenerate synth for the cluster's configs, eyeball side-by-side, then
confirm the cluster's KS/C2ST moved the right way without breaking learnability.

---

## 3. Levers to explore (pursue ALL, in order, with a visual loop after each)

The maintainer's directive: **pursue every lever one after another until the synth reaches the
visual quality of the test set.** Order = biggest visible win first.

1. **Noise amplitude + color (cluster A/B first — biggest visual win).** Replace white / AR(1)
   jitter with a **smooth, low-frequency** stochastic component: integrated noise
   (Brownian/random-walk), an **OU process with a long *time* correlation length**, or low-pass
   (moving-average / Gaussian-smoothed) noise. **Cut the noise amplitude** hard — most configs
   need far less than we inject. This alone should remove the orange noise-floor.
2. **Per-config SNR / smoothness fit (coordinate descent, but visually gated).** Fit, per config,
   the noise **amplitude** and the **correlation length / smoothing bandwidth** (and the
   seasonal/trend/noise split) by coordinate descent — objective = feature-distance with extra
   weight on `acf1`/`acf_diff1`/`spectral_entropy`/`stationarity`. **Gate every fitted knob-set on
   the side-by-side plot** (anti-stat-hack). No raw test values enter the fit — only the aggregate
   profile (no leakage).
3. **Quiet-baseline spikes (cluster D).** Make the spike baseline genuinely silent (≈0 + tiny
   smooth noise), spikes sparse + sharp; keep them **regular** where the config is periodic
   (predictable), sparse-random otherwise. Today's `0.25×AR` baseline is too loud.
4. **Smooth deterministic envelopes (cluster A/C).** Add generators we currently lack: logistic /
   Gompertz growth (covid_deaths), Gaussian/raised-cosine **bumps** (bizitobs_service), smooth
   cubic-spline drifts through a few random knots. Tiny additive smooth noise.
5. **Out-of-window long cycle → smooth drift (cluster A fine-sampled).** When a config's dominant
   real-world period exceeds the window (e.g. bizitobs_l2c/5T period 2036 in a 500-pt window),
   **render it as a smooth partial-cycle drift**, not dropped-to-noise (current `2≤p≤n/2` clamp
   silently drops it).
6. **Continuous-time generation (the unifying refactor — once 1–5 validate the direction).**
   Parametrize by **(real-world periodicity set in time units) × (sampling interval)**; build a
   wall-clock signal (sinusoids at real periods + OU/GP with a *time* correlation length + smooth
   trend) and **sample at the config's Δ**. samples-per-cycle, local smoothness, and spectral
   concentration all emerge correctly. Subsumes levers 1 & 5 and the period×freq variety goal.
7. **Cross-frequency resampling augmentation (variety).** Decimate/interpolate one time-domain
   signal to manufacture the same pattern at several samples-per-cycle — cheap coverage of the
   {period}×{frequency} cross-product.
8. **Colored-noise / PSD-slope matching (aperiodic-smooth).** Fit a 1/f^β envelope per config so
   `spectral_entropy` matches for configs with no clean discrete period (cluster C/E smooth-ish).

---

## 4. The visual-inspection workflow loop (run every iteration)

Per [[visual-first-synth-quality]]: **the plot is the primary gate; stats only confirm.**

1. Pick a cluster (or a single config for a hard case).
2. Generate a side-by-side panel: N real exemplars (blue) vs N current synth (orange) for each
   config in the cluster, **windowed to the same length** and spanning ≥2 cycles.
3. **Eyeball first** — does synth match: smoothness / HF-SNR? baseline quietness? envelope shape?
   spike regularity? trend? Note specific mismatches in words.
4. Apply the lever, regenerate, re-plot. Keep iterating until the panel *looks* like real.
5. **Only then** check the stats moved with the picture: per-config KS on
   `acf1`/`acf_diff1`/`spectral_entropy`/`stationarity` down toward / below general; dynamics-C2ST
   (kNN) down; learnability still ≈ real (1.27), not trivially easy.
6. Record the per-config scorecard row (KS deltas + a visual thumbnail) so a number can never move
   without the picture. Commit the refreshed panel as the standard artifact.

**Make the loop fast:** a committed CLI that regenerates the per-cluster panel + the per-config
scorecard in one shot (replaces the throwaway script that made the diagnostic). Likely
`src/tetris/data/synth_visual.py` (matplotlib Agg) + a `per_config_scorecard` in
`quality_harness.py`.

---

## 5. Tier-1 pass bar (combination gate; visual sign-off is PRIMARY)

The data is "good enough" to move to Tier-2/training when **all** hold:
- **Visual sign-off** per cluster — the side-by-side panel looks like real (primary, non-negotiable).
- **dynamics-C2ST (kNN)** for targeted below a set threshold (and clearly below general).
- **per-config smoothness KS** (`acf1`/`spectral_entropy`/`stationarity`) down to ≈ general or
  below, for the large majority of configs.
- **learnability** within X of real (≈1.27), not trivially easy (don't overshoot into too-smooth).
- **coverage**: all-but-a-named-few configs clear their own per-config bar (scorecard).

---

## 6. Implementation notes (files / scope, for the build session)

- `data/synthetic_targeted.py` — the generator rebuild: smooth low-freq noise component
  (`_smooth_noise` / OU-with-time-correlation), per-config noise-amplitude + correlation-length
  knobs (`TargetedKnobs`), quiet-baseline spikes, smooth-envelope generators, out-of-window drift.
  Keep defaults byte-compatible so existing tests pass; new behavior behind the fitted knobs.
- `data/synth_fit.py` (new) — per-config **coordinate-descent** knob fitting against the aggregate
  profile (no leakage); writes `targeted_knobs` into the profile JSON.
- `data/features.py` — possibly a richer smoothness descriptor (acf at a few lags / PSD slope) so
  the fit has the right target; a `samples_per_cycle` / calendar-period helper for the time×freq lens.
- `data/quality_harness.py` — `per_config_scorecard` (C2ST + learnability + smoothness-KS per
  config) and the **combination pass-bar gate**; keep kNN as the honest headline (no new classifier).
- `data/synth_visual.py` (new) — committed per-cluster real-vs-synth panel CLI (replaces throwaway).
- Tests: knob-fit determinism + reduces feature-distance + no-leakage guard; smooth-noise acf
  ordering; quiet-baseline spike; smooth-envelope shapes; scorecard + gate logic; period×freq helper.
- **Out of scope** (unchanged): the H2 curriculum/30M/clean-target-loss; `pack`/`assemble`/model;
  the frozen fixed-window seam.

## 7. Open questions to resolve while building
- OU/GP correlation length: fit per config from the profile's acf1, or search? (start: derive a
  seed from acf1, refine by coordinate descent.)
- Do we bake one generator per config (stable) or keep per-series goodness-of-fit selection
  (variety)? Lean: fitted recipe per cluster/config + per-series randomization of phase/length/knots.
- How much noise is "right" — fit absolute amplitude vs. SNR ratio? (SNR ratio is more transferable
  across scales.)
- Continuous-time refactor (lever 6): do it once levers 1–5 prove the direction, or commit to it
  up front as the clean foundation? (Captured for the build session to decide with fresh eyes.)
