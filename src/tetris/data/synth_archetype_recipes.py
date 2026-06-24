"""Recipes + a variety sampler over the H1.1 learnable archetypes — *how to actually
generate data* with :mod:`tetris.data.synth_archetypes`.

Two entry points:

* :func:`gen_from_recipe` — reproduce a *specific* characterized config (solar, bizitobs,
  electricity, covid, jena) from its validated per-channel ``(archetype, params)`` spec
  (the **targeted** use). The recipes are the hand-validated H1.1 parameters; the committed
  characterizer (next step) will eventually derive these per config from the real data.
* :func:`gen_variety` — sample the **archetype × params × period × sampling-interval**
  cross-product to manufacture a *wide variety* of learnable series (the **general** use).
  Same generators, assignment from sampled proportions instead of a real config.

Run ``python -m tetris.data.synth_archetype_recipes`` to write a preview PNG + an ``.npz``
of varied series you can inspect / train on.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import synth_archetypes as A

# Common sampling intervals (minutes) and pattern periods (minutes) GIFT-Eval spans.
INTERVALS_MIN = (1, 5, 10, 15, 60, 1440)          # 1T,5T,10T,15T,H,D
PERIODS_MIN = (720, 1440, 10080)                  # 12-hour, daily, weekly


# --- validated per-config recipes (H1.1) -------------------------------------
# Each recipe: period of the daily cycle in minutes + a list of per-channel
# (archetype, params). Multivariate configs list all channels; univariate list one.

def _jena_channels() -> List[Tuple[str, dict]]:
    spec: List[Optional[Tuple[str, dict]]] = [None] * 21
    weekly = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19]; pulse = [16, 17, 18]
    drift = [0, 13]; noisy = [11, 12, 14]; persist = [15, 20]
    for c in weekly:  spec[c] = ("drift_seasonal", dict(weekly_amp=1.0, drift_corr_days=10, noise_amp=0.25))
    for c in pulse:   spec[c] = ("recurring", dict(kind="pulse", weekly=False, amp_jitter=0.25, amp_persist=0.5, noise_amp=0.02, mult_noise=0.6))
    for c in drift:   spec[c] = ("drift_seasonal", dict(weekly_amp=0.0, drift_corr_days=6, noise_amp=0.12))
    for c in noisy:   spec[c] = ("drift_seasonal", dict(weekly_amp=0.2, drift_corr_days=8, noise_amp=0.7))
    for c in persist: spec[c] = ("drift_seasonal", dict(weekly_amp=0.3, drift_corr_days=8, noise_amp=0.15))
    return spec  # type: ignore[return-value]


def _ett_channels() -> List[Tuple[str, dict]]:
    # ETT (electricity transformer) = 7 heterogeneous channels that partially co-move
    # (load features + oil temperature), like jena. Almost all are a slow multi-month
    # drift backbone + a rounded daily cycle + noise (drift_seasonal); one load channel is
    # a blocky on/off load-switch (business + regime); the oil-temp channel (last) is a
    # near-pure smooth drift with no daily cycle.
    # daily cycle COMPARABLE to the drift (real ETT shows both a clear daily cycle AND a
    # slow drift), with a *shorter* drift correlation so the drift doesn't become one giant
    # swing that crushes the daily at fine sampling, + WHITE hf_noise (real channels have
    # genuine sample-to-sample jitter; was reading too clean). Per-day amp jitter keeps the
    # daily from being a constant-amplitude sine.
    daily = dict(weekly_amp=0.0, daily_amp=0.7, drift_corr_days=7.0, noise_amp=0.1, hf_noise=0.12)
    daily_noisy = dict(weekly_amp=0.0, daily_amp=0.62, drift_corr_days=6.0, noise_amp=0.14,
                       hf_noise=0.14)
    return [
        ("drift_seasonal", dict(daily)),                  # ch0: drift + clear daily
        ("drift_seasonal", dict(daily_noisy)),            # ch1: noisier daily
        ("drift_seasonal", dict(daily)),                  # ch2: drift + clear daily
        ("drift_seasonal", dict(daily_noisy)),            # ch3: noisier daily
        ("drift_seasonal", dict(weekly_amp=0.0, daily_amp=0.45, drift_corr_days=9.0,
                                noise_amp=0.15, hf_noise=0.14)),  # ch4: drift + moderate daily
        ("recurring", dict(kind="business", weekly=False, amp_persist=0.85, amp_jitter=0.2,
                           noise_amp=0.12, hf_noise=0.1, regime_prob=0.06,
                           regime_quiet=(0.05, 0.2))),      # ch5: blocky load-switch
        ("drift_seasonal", dict(weekly_amp=0.0, daily_amp=0.06, drift_corr_days=14.0,
                                noise_amp=0.05, hf_noise=0.04)),  # ch6: oil-temp, near-pure drift
    ]


RECIPES: Dict[str, dict] = {
    "solar": {"period_min": 1440, "tie": 0.0, "channels": [
        ("recurring", dict(kind="pulse", weekly=False, amp_jitter=0.12, amp_persist=0.7,
                           noise_amp=0.02, mult_noise=0.5))]},
    "bizitobs": {"period_min": 1440, "tie": 0.0, "channels": [
        ("recurring", dict(kind="business", weekly=False, amp_jitter=0.07, amp_persist=0.85,
                           noise_amp=0.03))]},
    "electricity": {"period_min": 1440, "tie": 0.0, "channels": [
        ("recurring", dict(kind="double_hump", weekly=True, amp_jitter=0.30, amp_persist=0.55,
                           noise_amp=0.08, hf_noise=0.06, regime_prob=0.02))]},
    "covid": {"period_min": 1440, "tie": 0.0, "channels": [
        ("growth", dict(kind="logistic"))]},
    "jena": {"period_min": 1440, "tie": 0.3, "channels": _jena_channels()},
    # ETT (ett1/ett2): 7 heterogeneous transformer channels (load + oil temp) that
    # partially co-move — drift+daily, a blocky load-switch, and a near-pure-drift OT.
    "ett": {"period_min": 1440, "tie": 0.3, "channels": _ett_channels()},
    # traffic FLOW (M_DENSE): clean daily cycle rising from a low night floor to a BROAD
    # daytime plateau (~16h elevated, short night), weekday/weekend contrast. The broad
    # day/short-night proportion is the dominant learnable feature → `business`. snaive ≪
    # last on the real.
    "traffic_flow": {"period_min": 1440, "tie": 0.0, "channels": [
        ("recurring", dict(kind="broad_hump", weekly=True, amp_jitter=0.18, amp_persist=0.6,
                           noise_amp=0.06, hf_noise=0.09))]},
    # taxi DEMAND (SZ_TAXI): TWO regimes — a cyclic stretch (strong daily wave, high day +
    # sharp pre-dawn troughs) then a flat noise-dominated stretch — with heavy noise
    # throughout. The `regime` envelope scales the daily profile down to a quiet level
    # while the residual stays constant, so a quiet stretch reads as "flat + noise" and an
    # active stretch as "cycle + noise" — the real two-regime structure. snaive only just
    # beats last (the cycle is weak relative to the noise), as on the real.
    "taxi_demand": {"period_min": 1440, "tie": 0.0, "channels": [
        ("recurring", dict(kind="broad_hump", weekly=False, amp_jitter=0.25, amp_persist=0.85,
                           noise_amp=0.12, hf_noise=0.15, regime_prob=0.05,
                           regime_quiet=(0.04, 0.15)))]},
    # traffic SPEED (LOOP_SEATTLE): high free-flow plateau notched DOWN by two rush-hour
    # dips/day (the `valley` profile) + heavy HF jitter (speed is noisy at 5T/H).
    "traffic_speed": {"period_min": 1440, "tie": 0.0, "channels": [
        ("recurring", dict(kind="valley", weekly=True, amp_jitter=0.10, amp_persist=0.85,
                           noise_amp=0.04, hf_noise=0.09))]},
    # --- M4 (univariate competition series; trend/growth-dominated, freq-dependent season) ---
    # m4_hourly: a clean, smooth DAILY cycle (call interval_min=60 ⇒ spc 24) on a slight
    # trend, low noise — high-SNR. snaive ≪ last on the real.
    "m4_hourly": {"period_min": 1440, "tie": 0.0, "channels": [
        ("drift_seasonal", dict(daily_amp=1.3, weekly_amp=0.0, drift_corr_days=10.0,
                                noise_amp=0.05, hf_noise=0.03, trend=1.0))]},
    # m4_annual: an ANNUAL cycle + slow drift + trend. One recipe serves m4_monthly
    # (call interval_min=43800 ⇒ spc 12) and m4_quarterly (interval_min=131400 ⇒ spc 4) via
    # the period×sampling decoupling. Seasonality is secondary to the trend (as on the real,
    # where linear/last-value edges seasonal-naive).
    "m4_annual": {"period_min": 525600, "tie": 0.0, "channels": [
        ("drift_seasonal", dict(daily_amp=0.55, daily_amp_jitter=0.7, weekly_amp=0.0,
                                drift_corr_days=4.0, noise_amp=0.22, hf_noise=0.16,
                                trend=0.8))]},
    # m4_trend: NON-seasonal trend + random-walk wander (m4_daily / m4_quarterly / m4_yearly) —
    # a persistent linear trend + the mean-reverting smooth drift, so last-value/linear are
    # forecastable but the wander makes pure-linear lose (matches the real). (M4 daily/
    # quarterly/yearly are mostly smooth trended curves with a turning point.)
    "m4_trend": {"period_min": 1440, "tie": 0.0, "channels": [
        ("drift_seasonal", dict(daily_amp=0.0, weekly_amp=0.0, drift_corr_days=4.0,
                                noise_amp=0.04, trend=2.2))]},
    # m4_spiky: regular SEASONAL spikes — the spiky-seasonal m4_weekly plurality (quiet
    # baseline + sharp peaks recurring at the ~annual-at-weekly period). Call interval_min=
    # 10080 ⇒ spc 52 (annual cycle at weekly sampling). One spike per period at a consistent
    # phase ⇒ seasonal-naive anticipates it (learnable, not random). (Weekly also has a
    # smooth-growth subtype → use m4_trend with a strong trend for that minority.)
    "m4_spiky": {"period_min": 525600, "tie": 0.0, "channels": [
        ("spikes", dict(rate_per_day=0.6, amp=4.0, weekly_amp=0.0, noise_amp=0.1))]},
    # --- counts / retail (univariate; overdispersed non-negative integer counts) ----------
    # restaurant: noisy DAILY visit counts (≈0–60 around a mean ~20), no clean seasonality —
    # a slowly-wandering level with heavy overdispersed count noise on top (last-value ≈
    # linear on the real; the noise is the dominant, irreducible feature). period=weekly so
    # the level-wander correlation is a few weeks at daily sampling.
    "restaurant": {"period_min": 10080, "tie": 0.0, "channels": [
        ("counts", dict(level=20.0, dispersion=0.4, level_drift=0.3, level_corr_days=8.0))]},
    # hospital: MONTHLY admission counts (≈0–27), weak yearly season + slow level shifts
    # (a drop-then-recover over the 6-year history). linear/last edge seasonal-naive on the
    # real → a wandering level (higher drift, shorter correlation so it shifts within the
    # record) dominates a small repeating yearly modulation. spc=12 at monthly sampling.
    "hospital": {"period_min": 525600, "tie": 0.0, "channels": [
        ("counts", dict(level=13.0, dispersion=0.16, level_drift=0.1, level_corr_days=2.2,
                        shift_amp=0.6, shift_corr_days=2.2, season_amp=0.08))]},
    # car_parts_with_missing: INTERMITTENT demand — mostly exact zeros with rare small
    # integer demands (1–2). A very low intensity + strong zero-inflation; no smooth
    # generator produces this discrete sparse-count texture. spc=12 (monthly/yearly).
    "car_parts": {"period_min": 525600, "tie": 0.0, "channels": [
        ("counts", dict(level=0.45, dispersion=0.6, level_drift=0.4, level_corr_days=2.0,
                        intermittent=0.5))]},
    # hierarchical_sales (D and W): a QUIET low count baseline punctuated by RARE, GIANT
    # spikes (baseline ~few, spikes 20–30×), aperiodic (season reported as 1) →
    # sparse-random spikes, not a recurring profile. The defining property is **low
    # predictability**: the real last-/linear-MASE is 2.6–3.6 because a rare giant spike
    # landing in the horizon wrecks any forecast — so the synth must be hard too (quiet
    # baseline so spikes dominate the scale, spikes rare + tall, *not* a busy easy baseline).
    # One recipe serves D (interval 1440 ⇒ spc 7) and W (interval 10080 ⇒ spc ~1→clamped).
    "hierarchical_sales": {"period_min": 10080, "tie": 0.0, "channels": [
        ("counts", dict(level=4.0, dispersion=0.4, level_drift=0.2, level_corr_days=6.0,
                        spike_rate=0.018, spike_amp=28.0))]},
    # --- remainder batch (kdd / temperature_rain / solar-D-W / electricity-D-W / bitbrains) --
    # kdd_cup_2018 (air-quality / pollution concentration): a non-negative LOW baseline (~50)
    # with broad EPISODIC bursts (plumes that rise + fall over many hours at H, sharp spikes
    # at D) and heavy overdispersion — poorly predictable (real snaive≈last≈lin≈3.8–4.0). Not
    # counts literally, but the texture is exactly the overdispersed non-negative bursty shape
    # gen_counts makes: a fast-wandering intensity (short level_corr ⇒ the bursts) + sparse tall
    # plumes. One recipe serves H (interval 60 ⇒ spc 24, ~half-day bursts) and D (interval 1440
    # ⇒ spc clamped, sharper). The gate is PREDICTABILITY PARITY (it must stay hard), not learnable.
    "kdd_pollution": {"period_min": 1440, "tie": 0.0, "channels": [
        ("counts", dict(level=50.0, dispersion=0.6, level_drift=1.3, level_corr_days=0.5,
                        spike_rate=0.012, spike_amp=5.0))]},
    # temperature_rain is a MIXED config (to_univariate flattens temperature AND rain series),
    # so it needs two recipes (like M4's per-subtype recipes):
    #   * temperature — a noisy slow WANDER (broad multi-month humps = the annual swing seen
    #     over <2 cycles) with heavy day-to-day jitter; aperiodic (m=1), lin≈last (~1.1) since
    #     the local drift gives a slope. drift_seasonal backbone (no season) + large noise/hf.
    "temperature": {"period_min": 1440, "tie": 0.0, "channels": [
        ("drift_seasonal", dict(daily_amp=0.0, weekly_amp=0.0, drift_corr_days=8.0,
                                noise_amp=0.35, hf_noise=0.18))]},
    #   * rain — INTERMITTENT precipitation with VOLATILITY CLUSTERING: alternating WET periods
    #     (frequent large bursts) and DRY periods (long calm zero stretches) — *not* uniformly
    #     random bursts (maintainer). Driven by a strong, slow intensity wander (large level_drift
    #     + multi-period corr) so λ swings between wet (big overdispersed counts) and dry (zeros);
    #     the bursts come from λ, not a uniform spike rate. MASE is DEGENERATE here (near-all-zero
    #     ⇒ naive-diff scale collapses, like car_parts) so the VISUAL wet/dry clustering is the gate.
    "rain": {"period_min": 1440, "tie": 0.0, "channels": [
        ("counts", dict(level=2.5, dispersion=1.8, level_drift=2.0, level_corr_days=6.0,
                        intermittent=0.7, spike_rate=0.006, spike_amp=22.0))]},
    # solar-D / solar-W are coarse re-samplings of solar: the intra-day pulse is gone; what
    # remains is the ANNUAL production envelope (summer high, winter low) seen over <2 cycles —
    # a slow rise-peak-fall hump. D carries heavy day-to-day CLOUD variability; W (aggregated)
    # is smooth. Composed from drift_seasonal's slow-drift backbone (the hump) — two recipes
    # since the noise level differs sharply (real last≈lin≈0.99 at D; last 1.74 < lin 2.56 at W,
    # a mean-reverting hump). spc set so the drift correlation spans the window.
    # The annual envelope is rendered as a partial-cycle SINE (daily_amp at the annual spc) —
    # a sine always swings (a long-correlation random drift can land flat in a <1-cycle window,
    # which earlier gave a degenerate flat panel). drift_corr_days is set LONG so the drift is a
    # near-flat offset and the sine envelope dominates. D carries heavy cloud noise; W is smooth.
    "solar_daily": {"period_min": 525600, "tie": 0.0, "channels": [
        ("drift_seasonal", dict(daily_amp=1.3, daily_amp_jitter=0.2, weekly_amp=0.0,
                                drift_corr_days=2.0, noise_amp=0.4, hf_noise=0.28))]},
    "solar_weekly": {"period_min": 525600, "tie": 0.0, "channels": [
        ("drift_seasonal", dict(daily_amp=1.3, daily_amp_jitter=0.15, weekly_amp=0.0,
                                drift_corr_days=2.0, noise_amp=0.16, hf_noise=0.07))]},
    # electricity-D / electricity-W are coarse re-samplings of electricity: the intraday
    # double-hump is gone; what remains is REGIME structure that is **trapezoidal** (a block
    # RISES to a high level, STAYS for a stretch with HF noise, then FALLS back) over a near-flat
    # low baseline — *not* abrupt random level shifts (maintainer). Modeled as a `recurring`
    # **business** profile (the rise/stay/fall trapezoid) at a multi-period BLOCK period, with
    # per-block amplitude variation (tall ↔ short blocks) and `regime_quiet` stretches that scale
    # quiet runs down to a near-flat low baseline, plus HF noise on the plateaus. A tall narrow
    # block reads as a "spike" at full scale (lesson: a tall narrow trapezoid looks like a spike).
    #   * electricity_weekly — blocks ~30 weeks (period 302400 min ⇒ spc 30 @W); held high
    #     plateaus alternating with flat low baseline (real last 1.05 ≪ lin 2.54: last-value
    #     forecasts within a held block, linear overshoots at the edges).
    "electricity_weekly": {"period_min": 302400, "tie": 0.0, "channels": [
        ("recurring", dict(kind="business", weekly=False, amp_jitter=0.4, amp_persist=0.7,
                           noise_amp=0.04, hf_noise=0.12, regime_prob=0.16,
                           regime_quiet=(0.02, 0.1)))]},
    #   * electricity_daily — shorter blocks (~20 days, period 28800 ⇒ spc 20 @D) with HIGH
    #     per-block amplitude variation (low persistence) so occasional tall narrow blocks read
    #     as the sparse tall spikes seen on the quiet baseline — very hard (real last≈lin≈4.2:
    #     a tall block landing in the horizon wrecks any forecast, predictability-parity gate).
    "electricity_daily": {"period_min": 43200, "tie": 0.0, "channels": [
        ("recurring", dict(kind="business", weekly=False, amp_jitter=0.7, amp_persist=0.5,
                           noise_amp=0.05, hf_noise=0.15, regime_prob=0.16,
                           regime_quiet=(0.03, 0.12)))]},
    # bitbrains_fast_storage (H + 5T): a QUIET noisy baseline (~700) with sparse SHARP spikes
    # (rare giants to several× baseline), two co-moving channels. Exactly the overdispersed
    # non-negative quiet-baseline + sparse-spike shape — gen_counts at a high level (Poisson
    # noise gives the ±few-% baseline jitter) + spike_rate. Aperiodic spikes (5T last 1.25 <
    # snaive 1.60); H weakly daily (snaive 0.654 < last 0.675). One recipe for both freqs.
    "bitbrains": {"period_min": 1440, "tie": 0.0, "channels": [
        ("counts", dict(level=700.0, dispersion=0.012, level_drift=0.04, level_corr_days=1.0,
                        season_amp=0.08, spike_rate=0.003, spike_amp=4.0))]},
    # --- river / births batch (saugeenday + us_births, D/W/M) ------------------------------
    # saugeenday = RIVER FLOW: a low non-negative baseline punctuated by sharp asymmetric FLOOD
    # events — a fast rise then a slow exponential RECESSION (the hydrograph shape), aperiodic
    # at D/W (weather-driven), clustering into an annual spring freshet at M. gen_counts with the
    # NEW `spike_decay` recession. D and W differ only in the recession length in SAMPLES (the
    # event lasts ~3 weeks ⇒ ~18 samples at D, ~3 at W) ⇒ two recipes. Real last < lin (mean-
    # reverting to baseline): a flood spike landing in the horizon is unpredictable (hard).
    # the floods recur with an ANNUAL rhythm (spike_season_spc — 365 @D / 52 @W) and a slow
    # amplitude wave (bigger spring freshets), not uniformly at random (maintainer); each event
    # keeps the asymmetric hydrograph recession.
    "saugeenday_daily": {"period_min": 1440, "tie": 0.0, "channels": [
        ("counts", dict(level=40.0, dispersion=0.12, level_drift=0.5, level_corr_days=2.0,
                        spike_rate=0.012, spike_amp=7.0, spike_decay=15.0, spike_season_spc=365.0))]},
    "saugeenday_weekly": {"period_min": 1440, "tie": 0.0, "channels": [
        ("counts", dict(level=40.0, dispersion=0.12, level_drift=0.5, level_corr_days=4.0,
                        spike_rate=0.035, spike_amp=7.0, spike_decay=3.0, spike_season_spc=52.0))]},
    # saugeenday_monthly — the monthly aggregate reveals a strong ANNUAL freshet cycle (real
    # snaive 0.31 ≪ last 0.86): a sharp spring peak repeating yearly on a low baseline. A
    # `recurring` single_hump at the annual period (spc 12 @M) is the repeating peak (seasonal-
    # naive learnable); mult/hf noise gives the year-to-year + within-peak variation.
    "saugeenday_monthly": {"period_min": 525600, "tie": 0.0, "channels": [
        ("recurring", dict(kind="single_hump", weekly=False, amp_jitter=0.35, amp_persist=0.4,
                           noise_amp=0.05, hf_noise=0.05, mult_noise=0.15))]},
    # us_births = BIRTH COUNTS. Daily has a strong WEEKLY cycle (weekday-high / weekend-low —
    # scheduled deliveries) + a slow multi-year drift. The weekly swing is sizeable and the
    # pattern is REGULAR/predictable (maintainer) ⇒ a `business` weekly profile (spc 7 @D) with a
    # deeper swing (lower level_frac) and a clean low-noise cycle; the persistent `level` still
    # carries the slow drift. snaive-at-7 beats last (the weekly profile repeats).
    "us_births_daily": {"period_min": 10080, "tie": 0.0, "channels": [
        ("recurring", dict(kind="business", weekly=False, amp_jitter=0.08, amp_persist=0.9,
                           noise_amp=0.05, hf_noise=0.03, level_frac=0.45))]},
    # us_births W — a FIXED annual pattern repeating regularly (a noisy floor, a sudden rise to a
    # peak, a fall back to the floor) whose peak height WAVES slowly over the years (an overall
    # low-freq multi-year envelope), maintainer. A `recurring` single_hump at the ANNUAL period
    # (spc 52 @W) gives the repeating floor→rise→peak→fall pattern; high `amp_persist` makes the
    # per-YEAR amplitude an AR(1) that wanders slowly ⇒ the long-term wave; mult/hf noise give the
    # noisy floor + year-to-year randomness.
    "us_births_weekly": {"period_min": 525600, "tie": 0.0, "channels": [
        ("recurring", dict(kind="broad_hump", weekly=False, amp_jitter=0.3, amp_persist=0.85,
                           noise_amp=0.22, hf_noise=0.1, mult_noise=0.3, level_frac=0.45))]},
    # us_births M — a rounded ANNUAL cycle + multi-year drift (drift_seasonal annual `daily_amp`
    # sine). Real M is strongly seasonal (snaive 0.30 ≪ last) but NOT a clean sine — it carries
    # real month-to-month noise (maintainer: "we need noise"), so a sizeable residual on top.
    "us_births_monthly": {"period_min": 525600, "tie": 0.0, "channels": [
        ("drift_seasonal", dict(daily_amp=1.6, daily_amp_jitter=0.15, weekly_amp=0.0,
                                drift_corr_days=4.0, noise_amp=0.18, hf_noise=0.09))]},
}


def gen_from_recipe(rng, name: str, n: int, *, interval_min: int = 10) -> np.ndarray:
    """Generate ``[C, n]`` for a characterized config from its validated recipe. The daily
    period is rendered at ``interval_min`` (so ``samples_per_cycle = period/interval`` —
    e.g. solar at 10-min ⇒ 144, at hourly ⇒ 24)."""
    rec = RECIPES[name]
    spc = A.samples_per_cycle(rec["period_min"], interval_min)
    spc = int(np.clip(spc, 4, max(4, n // 3)))
    return A.gen_multivariate(rng, n, spc, rec["channels"], tie=rec.get("tie", 0.0))


# --- variety sampler (the general family) ------------------------------------

def gen_variety(rng, n: int) -> Tuple[np.ndarray, dict]:
    """Sample one varied learnable series ``[C, n]`` from the archetype × params × period ×
    sampling cross-product. Returns ``(data, meta)``; ``meta`` records the draw so you can
    bin/inspect. Every draw is learnable by construction (a repeating profile, a clean
    growth, or a persistent drift + cycle)."""
    interval = int(rng.choice(INTERVALS_MIN))
    period = int(rng.choice(PERIODS_MIN))
    spc = int(np.clip(A.samples_per_cycle(period, interval), 4, max(4, n // 3)))
    family = rng.choice(["recurring", "growth", "drift_seasonal", "spikes", "multivariate"],
                        p=[0.4, 0.1, 0.25, 0.1, 0.15])

    def recurring_params():
        return dict(
            kind=str(rng.choice(A.PROFILE_KINDS)),
            weekly=bool(rng.random() < 0.5),
            amp_jitter=float(rng.uniform(0.05, 0.35)),
            amp_persist=float(rng.uniform(0.4, 0.92)),
            noise_amp=float(rng.uniform(0.02, 0.2)),
            mult_noise=float(rng.choice([0.0, rng.uniform(0.2, 0.6)])),
            hf_noise=float(rng.choice([0.0, rng.uniform(0.05, 0.3)])),
            level_frac=float(rng.choice([0.0, rng.uniform(0.3, 0.7)])),
            regime_prob=float(rng.choice([0.0, rng.uniform(0.01, 0.04)])),
        )

    if family == "growth":
        data = A.gen_growth(rng, n, kind=str(rng.choice(A.GROWTH_KINDS)))[0][None, :]
    elif family == "drift_seasonal":
        data = A.gen_drift_seasonal(rng, n, spc, weekly_amp=float(rng.choice([0.0, rng.uniform(0.4, 1.2)])),
                                    daily_amp=float(rng.choice([0.0, rng.uniform(0.1, 0.5)])),
                                    drift_corr_days=float(rng.uniform(4, 16)),
                                    noise_amp=float(rng.uniform(0.05, 0.6)))[0][None, :]
    elif family == "spikes":
        data = A.gen_channel(rng, n, spc, "spikes",
                             dict(rate_per_day=float(rng.uniform(0.1, 1.0)),
                                  amp=float(rng.uniform(2, 6)), noise_amp=0.03))[None, :]
    elif family == "multivariate":
        C = int(rng.integers(2, 8))
        archs = ["recurring", "drift_seasonal", "spikes"]
        specs = []                                        # one sensible spec per channel
        for _ in range(C):
            a = str(rng.choice(archs))
            p = recurring_params() if a == "recurring" else (
                dict(weekly_amp=float(rng.uniform(0, 1.0)), drift_corr_days=float(rng.uniform(4, 14)),
                     noise_amp=float(rng.uniform(0.1, 0.5))) if a == "drift_seasonal" else
                dict(rate_per_day=float(rng.uniform(0.1, 1.0)), amp=float(rng.uniform(2, 6))))
            specs.append((a, p))
        data = A.gen_multivariate(rng, n, spc, specs, tie=float(rng.uniform(0.0, 0.5)))
    else:                                                  # recurring (univariate)
        data = A.gen_recurring_profile(rng, n, spc, **recurring_params())[0][None, :]

    return data.astype(np.float64), {"family": family, "interval_min": interval,
                                     "period_min": period, "spc": spc, "C": data.shape[0]}


def write_archetype_corpus(writer, *, n_series: int, seed: int = 0,
                           length_range: Tuple[int, int] = (512, 4096),
                           source: str = "synth_archetype") -> int:
    """Generate ``n_series`` varied learnable-archetype series and feed them to a
    ``ShardWriter`` (same interface as ``write_general_corpus``). All channels are
    targets (``nf=0``); ``season_length`` is the samples-per-cycle. Deterministic:
    series ``idx`` keyed by ``(seed, marker, idx)``."""
    import numpy as _np
    lo, hi = int(length_range[0]), int(length_range[1])
    for idx in range(int(n_series)):
        rng = _np.random.default_rng((int(seed), 0xA3C7, idx))
        n = int(rng.integers(lo, hi + 1))
        data, meta = gen_variety(rng, n)
        data = _np.ascontiguousarray(_np.atleast_2d(data), dtype=_np.float32)
        C = data.shape[0]
        writer.add(data, 0, C, season_length=int(meta["spc"]), source=source,
                   kind=str(meta["family"]), item_id=f"arch_{idx}")
    return int(n_series)


def main() -> None:  # pragma: no cover - manual entrypoint / produces artifacts
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Generate a varied learnable-archetype synth set")
    ap.add_argument("--n-series", type=int, default=24)
    ap.add_argument("--length", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/synth_variety")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series, metas = [], []
    for i in range(args.n_series):
        d, meta = gen_variety(np.random.default_rng((args.seed, i)), args.length)
        series.append(d[0]); metas.append(meta)
    np.savez(args.out + ".npz", **{f"s{i}": s for i, s in enumerate(series)})
    cols = 4; rows = (len(series) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 1.5 * rows), squeeze=False)
    for i, (s, m) in enumerate(zip(series, metas)):
        a = ax[i // cols][i % cols]; a.plot(s[:min(len(s), 1500)], lw=0.6)
        a.set_title(f"{m['family']} {m['period_min']}m/{m['interval_min']}m spc={m['spc']}", fontsize=6)
        a.tick_params(labelsize=5)
    for j in range(len(series), rows * cols):
        ax[j // cols][j % cols].axis("off")
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=105); plt.close(fig)
    print(f"wrote {args.out}.npz and {args.out}.png ({len(series)} varied series)")


if __name__ == "__main__":  # pragma: no cover
    main()
