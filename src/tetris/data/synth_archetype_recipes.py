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
