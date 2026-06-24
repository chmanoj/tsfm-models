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
    # traffic FLOW (M_DENSE): clean daily cycle rising from a low night floor to a BROAD
    # daytime plateau (~16h elevated, short night), weekday/weekend contrast. The broad
    # day/short-night proportion is the dominant learnable feature → `business`. snaive ≪
    # last on the real.
    "traffic_flow": {"period_min": 1440, "tie": 0.0, "channels": [
        ("recurring", dict(kind="broad_hump", weekly=True, amp_jitter=0.18, amp_persist=0.6,
                           noise_amp=0.06, hf_noise=0.09))]},
    # taxi DEMAND (SZ_TAXI): a WEAK daily cycle drowned in heavy noise + occasional level
    # shifts — snaive only just beats last on the real (not a clean hump). Modeled as a
    # small daily oscillation on a persistent drift with a large residual.
    "taxi_demand": {"period_min": 1440, "tie": 0.0, "channels": [
        ("drift_seasonal", dict(weekly_amp=0.0, daily_amp=0.6, drift_corr_days=4.0,
                                noise_amp=0.7))]},
    # traffic SPEED (LOOP_SEATTLE): high free-flow plateau notched DOWN by two rush-hour
    # dips/day (the `valley` profile) + heavy HF jitter (speed is noisy at 5T/H).
    "traffic_speed": {"period_min": 1440, "tie": 0.0, "channels": [
        ("recurring", dict(kind="valley", weekly=True, amp_jitter=0.10, amp_persist=0.85,
                           noise_amp=0.04, hf_noise=0.09))]},
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
