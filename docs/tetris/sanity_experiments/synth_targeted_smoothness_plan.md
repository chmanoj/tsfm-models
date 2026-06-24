# Targeted-synth smoothness — plan of action (H1.1 working doc)

**Status (2026-06-17):** PIVOTED. The original "make synth smoother (fix acf1)" framing
below (sections 1–7) was partly a stat-hack trap — see the update section immediately
following. The work pivoted mid-session to **data-driven learnable archetypes**; the
generators are built and validated on the first datasets. Read this update section first,
then sections 1–7 for the original smoothness diagnosis (still useful background).
Principles: [[dont-game-synth-quality-metric]], [[synth-period-vs-sampling-frequency]],
[[visual-first-synth-quality]], [[learnable-structure-not-smooth-noise]].

---

## SESSION H1.1 UPDATE (2026-06-17) — pivot to data-driven learnable archetypes

### What happened
First attempt (committed `f4510e1`) chased the smoothness stat: a low-frequency
**smooth-random** backbone drove acf1 from a 0.40 gap to 0.07. The maintainer caught that
this **won the number but destroyed the learnable pattern** — the output looked flat /
sine-y and lost solar's daily pulse, covid's clean rise, bitbrains' recurring spikes. A
random walk has acf1≈1 yet is unforecastable — the exact [[dont-game-synth-quality-metric]]
trap. We **pivoted to data first**: look at the real series, find what's *learnable*
(seasonal-naive beats last-value), and build generators for those archetypes.

### The data-driven archetypes (built + validated — `src/tetris/data/synth_archetypes.py`)
Characterized solar, bitbrains, jena, bizitobs, covid, electricity (full-series,
multi-timescale, per-channel seasonal-naive). Almost every **learnable** pattern is a
**recurring profile**: a fixed within-period waveform that *repeats* (why seasonal-naive
forecasts it), with per-day amplitude variation, weekly modulation, small residual —
parametrized by **period-in-time × sampling interval** (solar daily = 144 smp@10T =
24 smp@H). Generators (all with tests, `tests/test_synth_archetypes.py`):
- `gen_recurring_profile` — **trapezoid** daily profiles (`_flat_top`: rise/stay/fall, flat
  plateau — *not* sine bells): `pulse` (solar, zero night), `business` (bizitobs), `double_hump`
  (electricity load), `single_hump`. Knobs: stay-length, edge taper, **HF cloud noise on the
  plateau** (`mult_noise`), persistent AR(1) day amplitude, weekly, `level_frac`, `hf_noise`,
  **active↔quiet `regime` shifts** (electricity busy/idle), `trend`.
- `gen_growth` — linear/logistic/exponential, kept **still-rising** at the horizon (covid).
- `gen_drift_seasonal` — the multi-scale weather/drift archetype (jena): slow long-correlation
  persistent backbone + optional weekly cycle + small daily ripple + noise.
- `gen_multivariate` — composer over per-channel `(archetype, params)` specs tied by a
  **shared seasonal envelope** (channels co-move). The parametrized superset: `tie=0` ⇒
  independent; assignment source = measured-per-config (targeted) vs sampled-proportions
  (general). Validated on jena (21 heterogeneous channels: weekly / solar-pulse / drift /
  turbulent; synth cross-corr 0.41 vs real 0.40).
- `samples_per_cycle`, `add_sparse_spikes`.

Validated (seasonal-naive synth≈real + visual): solar (trapezoid pulse + HF cloud),
bizitobs (trapezoid business profile), electricity (double-hump + day-variation + regime
shifts; snaive matches real 0.35), covid (growth), jena (archetype-level multivariate mix).
The visual loop CLI is `synth_visual.py`; throwaway exploration lived in `/tmp` (the
committed **characterizer** is a next step).

#### Batch progress (continuing the hand-characterization, 3–4 datasets per batch)
- **Traffic batch (2026-06-23) — DONE.** Characterized LOOP_SEATTLE, M_DENSE, SZ_TAXI
  (full-series, multi-scale, per-channel snaive-vs-last; all univariate, daily season).
  Three distinct learnable shapes, all snaive≪last on the real:
  - **LOOP_SEATTLE (speed)** → NEW `valley` profile: a high free-flow plateau notched
    *down* by sharp asymmetric rush-hour dips (the inverse of `double_hump`). Recipe
    `traffic_speed`. Validated cleanly at hourly; at 5T the real is near its noise floor
    (snaive≈last≈2.9) so synth is intentionally cleaner/more-learnable (don't chase noise).
  - **M_DENSE (flow)** → NEW `broad_hump` profile: a wide rounded daytime (~0.6 of the day,
    large taper) over a short low night — a soft trapezoid, *not* the square wave `business`
    gave nor the too-narrow `single_hump`. Recipe `traffic_flow`.
  - **SZ_TAXI (demand)** → **two regimes** (maintainer feedback): a cyclic stretch (rounded
    daily wave + noise) then a flat noise-dominated stretch. Modeled as a regime-switching
    `broad_hump` recurring profile — the `regime` envelope scales the daily profile down to
    a deep quiet level (new `regime_quiet` pass-through knob) while the residual stays
    constant, so a quiet stretch reads as *flat + noise* and an active stretch as *cycle +
    noise*. Recipe `taxi_demand`. (First pass used `drift_seasonal` → looked like uniform
    sine+noise with no regimes; the eye caught it.)
  Panels: `docs/tetris/sanity_experiments/synth_panels/h1_1_traffic/`. Subagent visual
  critique drove the `broad_hump` fix (M_DENSE square-wave) and confirmed the rest.
  New archetype vocabulary: `valley`, `broad_hump` (added to `PROFILE_KINDS`).

### How to generate data NOW (`src/tetris/data/synth_archetype_recipes.py`)
The validated recipes + a variety sampler are committed so the generators are usable
today (the per-config params won't be lost in throwaway scripts).

```python
import numpy as np
from tetris.data import synth_archetype_recipes as R

# (a) reproduce a characterized config from its validated recipe (the TARGETED use).
#     interval_min sets samples-per-cycle: solar daily = 144 @10-min, 24 @hourly.
solar  = R.gen_from_recipe(np.random.default_rng(0), "solar", n=4000, interval_min=10)  # [1, 4000]
jena   = R.gen_from_recipe(np.random.default_rng(0), "jena",  n=8000, interval_min=10)  # [21, 8000]
# names: solar, bizitobs, electricity, covid, jena   (RECIPES dict)

# (b) sample the archetype x params x period x sampling cross-product for WIDE VARIETY
#     (the GENERAL use) — every draw is learnable by construction.
for i in range(1000):
    data, meta = R.gen_variety(np.random.default_rng((seed, i)), n=4000)   # [C, 4000], meta = the draw
```
Or run the CLI to write a preview + an `.npz` you can inspect / train on:
```bash
uv run python -m tetris.data.synth_archetype_recipes --n-series 24 --length 4000 --out /tmp/synth_variety
```
Building blocks live in `synth_archetypes.py`: `gen_recurring_profile` (kinds
pulse/business/double_hump/single_hump), `gen_growth`, `gen_drift_seasonal`,
`gen_multivariate`, `add_sparse_spikes`, `samples_per_cycle`. Variety knobs: archetype/kind,
stay-length & edge taper, `mult_noise` (HF cloud on the plateau), `hf_noise`, `amp_jitter`/
`amp_persist`, `weekly`, `level_frac`, `regime_prob`, `drift_corr_days`, **period × sampling
interval**. (The recipes are hand-validated for 5 configs; the characterizer will later derive
them per config from the real data.)

### Lessons (maintainer feedback + mistakes I made — do not repeat)
1. **Don't stat-hack one metric.** Visual + learnability are co-primary; a smooth-random
   signal that wins acf1 is unlearnable. [[learnable-structure-not-smooth-noise]].
2. **Learnability signal = seasonal-naive beats last-value** (sampling-invariant), compared
   synth-vs-real *relatively* — NOT an absolute MASE threshold (fine sampling inflates MASE:
   solar/10T MASE 5.35 yet perfectly learnable).
3. **Plot the FULL series at multiple timescales** (full/month/week/day), never a fixed short
   window — I mislabelled jena from a 3-day zoom of a 1-year series.
4. **Look at ALL channels** (jena = 21 heterogeneous), not a subset.
5. **Real daily shapes are trapezoids (rise/stay/fall), not sine bells**; zoom in before
   calling something a "spike" (a tall narrow trapezoid looks like a spike at full scale).
6. **Trust the eye over stat-decompositions** — a phase-average misled me on electricity
   (it's a strong daily double-hump), a daily-only MASE mis-classified jena's spiky channels.

### Next steps (proceed methodically — 3–4 datasets per batch, like this session)
1. **Characterize the remaining datasets in batches** (full-series, multi-scale, per-channel
   seasonal-naive, eyeball): traffic (LOOP_SEATTLE / M_DENSE / SZ_TAXI); counts/retail
   (restaurant / car_parts / hospital / hierarchical_sales); river/births (saugeenday /
   us_births); ett / m4 / kdd / temperature_rain / solar-D-W / bitbrains-storage. Extend the
   archetype set only if a genuinely new shape appears.
2. **Build the committed characterizer** (`synth_explore.py` → formalize the `/tmp` tool):
   per config/channel, measure sampling interval, dominant period(s) **in time**, the
   learnability/archetype classification **by shape** (not MASE alone), and noise/amplitude —
   emit `(archetype, params)` per channel; store aggregate per-config specs in the profile
   (no leakage) for the targeted family.
3. **Wire archetypes into the corpus** — DONE for the variety family: `write_archetype_corpus`
   (`synth_archetype_recipes.py`) + `materialize --n-archetype N` write a `synth_archetype`
   shard corpus from `gen_variety`. Still TODO: a `gen_targeted` path that dispatches the
   archetype generators from stored per-config specs (after the characterizer).
   - **Immediate next: the 20M validation run (WSL).** Generate the corpus
     (`uv run python -m tetris.data.materialize --out outputs/corpus_archetype --n-archetype 50000`),
     then a config that **trains a ~20M model on that corpus (streaming loader)** and **evals
     zero-shot on the GIFT-Eval *test* split, 10 items/config** (both the 5 characterized
     datasets and all 97). Base the config on `configs/streaming_synth.yaml` (streaming loader
     on `corpus_archetype`) + the curriculum config's GIFT-Eval-test eval block; scale
     `model.d_model` to ~20M params. The eval wiring is the part to get right (a wrong eval
     block wastes a GPU run), so build it deliberately, not blind.
4. **Critic subagent + per-config learnability gate** — a visual-critique agent that grades
   real-vs-synth panels against the principles, plus a harness gate that requires
   seasonal-naive(synth) ≈ seasonal-naive(real) per config. A config passes only when visual +
   learnability + stats agree.
5. Then the Tier-1 scorecard / pass-bar, and on to Tier-2.

---

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
