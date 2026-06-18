"""Per-cluster real-vs-synth visual panels (H1.1) — *the primary Tier-1 gate*.

The maintainer's rule ([[visual-first-synth-quality]]): **the side-by-side plot is the
primary gate; stats only confirm.** This module replaces the throwaway script that made
``synth_smoothness_diagnostic.png`` with a committed, reproducible CLI: for each *dynamics
cluster* it draws N real GIFT-Eval exemplars (blue) next to N targeted-synth exemplars
(orange), windowed to the same length and spanning >=2 seasonal cycles, so a knob change
can never move a KS/C2ST number without the picture being checked against it.

Clusters group configs by *dynamics regime* (smoothness / impulsiveness), not frequency —
the axis the H1.1 smoothness work operates on (see the plan-of-action doc, section 2).

No leakage: real exemplars are only *rendered*, never fed back into generation; the synth
side reads the aggregate profile alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import features as F
from . import synthetic_targeted as TGT
from .test_profile import TestProfile

# Dynamics clusters (config keys are GIFT-Eval names incl. the freq suffix, matching the
# profile group keys). Mirrors the plan-of-action doc's A–E regimes; only configs that the
# fitted profile actually contains are listed.
CLUSTERS: Dict[str, List[str]] = {
    "A_ultra_smooth": [
        "jena_weather/10T", "bizitobs_l2c/5T", "solar/10T", "ett2/15T", "ett2/H",
        "bizitobs_application", "bizitobs_service", "m4_daily", "covid_deaths",
    ],
    "B_smooth_seasonal": [
        "electricity/15T", "LOOP_SEATTLE/5T", "solar/H", "kdd_cup_2018_with_missing/H",
        "M_DENSE/H", "m4_hourly", "m4_monthly", "ett1/15T", "ett1/H", "jena_weather/H",
    ],
    "C_trend_levelshift": [
        "electricity/D", "electricity/W", "m4_weekly", "m4_quarterly", "m4_yearly",
        "saugeenday/D", "saugeenday/W", "us_births/W",
    ],
    "D_impulsive": [
        "bitbrains_rnd/5T", "bitbrains_rnd/H", "bitbrains_fast_storage/5T",
        "bitbrains_fast_storage/H", "electricity/H", "temperature_rain_with_missing",
    ],
    "E_noisy_intermittent": [
        "restaurant", "car_parts_with_missing", "hospital", "hierarchical_sales/D",
        "hierarchical_sales/W", "solar/D", "us_births/D", "SZ_TAXI/15T",
    ],
}


def _window_len(season: int) -> int:
    """A render window spanning >=2 cycles, bounded so very long real series and the
    short synth series are compared on the same footing."""
    base = 3 * season if season and season >= 2 else 240
    return int(np.clip(base, 200, 600))


def _crop(rng, x: np.ndarray, win: int) -> np.ndarray:
    """A finite, length-``win`` window of ``x`` (mean-filled NaNs for display)."""
    x = F._fill_nan(np.asarray(x, dtype=np.float64).ravel())
    n = x.size
    if n <= win:
        return x
    start = int(rng.integers(0, n - win + 1))
    return x[start:start + win]


def real_exemplars(group: str, *, local_dir: str = "", n: int = 3, season: int = 0,
                   seed: int = 0) -> Tuple[List[np.ndarray], int]:
    """``n`` windowed real context series (channel 0) for ``group``; also returns the
    season length read from GIFT-Eval. Network/lazy (needs the test tree)."""
    from .gifteval_download import iter_eval_items

    rng = np.random.default_rng((seed, hash(group) & 0xFFFF))
    out: List[np.ndarray] = []
    found_season = season
    for ev in iter_eval_items(local_dir, configs=[group], items_per_config=max(n, 4),
                              terms=("short",)):
        found_season = int(ev.season_length) if ev.season_length else found_season
        data = ev.data_tensor.detach().cpu().numpy()
        out.append(data[0])
        if len(out) >= n:
            break
    win = _window_len(found_season)
    return [_crop(rng, s, win) for s in out], found_season


def synth_exemplars(group: str, profile: TestProfile, *, n: int = 3, season: int = 0,
                    seed: int = 0) -> List[np.ndarray]:
    """``n`` windowed targeted-synth series for ``group`` (channel 0)."""
    rng = np.random.default_rng((seed, hash(group) & 0xFFFF, 0x5))
    win = _window_len(season)
    out = []
    for i in range(n):
        g = group if group in profile.groups else None
        data, _nf, _nt, m, _grp = TGT.gen_targeted(
            np.random.default_rng((seed, hash(group) & 0xFFFF, i)), profile, group=g)
        out.append(_crop(rng, np.atleast_2d(data)[0], win))
    return out


def render_cluster_panel(cluster: str, profile: TestProfile, out_path, *,
                         local_dir: str = "", n_exemplars: int = 3, seed: int = 0) -> str:
    """Draw the real(blue)-vs-synth(orange) panel for one cluster → ``out_path``.

    Rows = configs in the cluster; left column = real exemplars, right column = synth,
    each cell overlaying ``n_exemplars`` z-scored windows so shape (not scale) is what the
    eye compares. Returns the written path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = [c for c in CLUSTERS[cluster] if c in profile.groups]
    nrows = len(configs)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 1.7 * nrows + 0.5), squeeze=False)
    fig.suptitle(f"cluster {cluster}: real (blue) vs targeted synth (orange)", fontsize=11)
    for r, cfg in enumerate(configs):
        reals, season = real_exemplars(cfg, local_dir=local_dir, n=n_exemplars, seed=seed)
        synths = synth_exemplars(cfg, profile, n=n_exemplars, season=season, seed=seed)
        for col, (series, color, tag) in enumerate(
                ((reals, "tab:blue", "real"), (synths, "tab:orange", "synth"))):
            ax = axes[r][col]
            for s in series:
                z = (s - np.mean(s)) / (np.std(s) + 1e-8)
                ax.plot(z, color=color, lw=0.8, alpha=0.8)
            ax.set_ylabel(cfg, fontsize=7) if col == 0 else None
            ax.set_title(f"{tag} (m={season})", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


def main() -> None:  # pragma: no cover - manual entrypoint / produces artifacts
    import argparse
    ap = argparse.ArgumentParser(description="Per-cluster real-vs-synth visual panels")
    ap.add_argument("--profile", required=True, help="committed TestProfile JSON")
    ap.add_argument("--local-dir", default="", help="GIFT-Eval test root (else $GIFT_EVAL)")
    ap.add_argument("--cluster", default="all",
                    help=f"one of {list(CLUSTERS)} or 'all'")
    ap.add_argument("--out-dir", default="docs/tetris/sanity_experiments/synth_panels")
    ap.add_argument("--n-exemplars", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    profile = TestProfile.load(args.profile)
    clusters = list(CLUSTERS) if args.cluster == "all" else [args.cluster]
    for cl in clusters:
        path = Path(args.out_dir) / f"panel_{cl}.png"
        render_cluster_panel(cl, profile, path, local_dir=args.local_dir,
                             n_exemplars=args.n_exemplars, seed=args.seed)
        print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
