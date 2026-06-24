"""Visual-first characterization + recipe-validation tooling (H1.1).

The committed version of the per-dataset workflow used to hand-characterize the GIFT-Eval
configs and validate the archetype recipes (replaces the throwaway scratchpad scripts).
**The plot is the primary gate; the MASE split only confirms the learnability direction**
([[visual-first-synth-quality]], [[dont-game-synth-quality-metric]]).

Three entry points (real data via ``$GIFT_EVAL`` / ``--local-dir``; no leakage — real
series are only *rendered*, never fed into generation):

* ``characterize`` — for one or more configs: print freq/season/channels/length + per-channel
  **seasonal-naive vs last-value vs linear** MASE (the learnability split: snaive<last ⇒ a
  learnable recurring pattern), and plot the **full series at multiple timescales** across
  **all channels**.
* ``panel`` — a univariate real(blue)-vs-synth(orange) panel for a ``(config, recipe,
  interval)`` at several zoom levels (full / ~8 cycles / ~3 cycles).
* ``panel-mv`` — a multichannel panel (rows = channels; REAL-8d / REAL-2d / SYNTH-8d /
  SYNTH-2d), for the composed multivariate recipes (ett, jena).

CLI::

    uv run python -m tetris.data.synth_explore characterize ett1/H ett2/H --n 3
    uv run python -m tetris.data.synth_explore panel LOOP_SEATTLE/H traffic_speed 60 --out p.png
    uv run python -m tetris.data.synth_explore panel-mv ett1/H ett 60 --out p.png
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import features as F
from . import synth_archetype_recipes as R


# --- learnability split (confirms the direction; the plot is the real gate) --------------

def mase_split(x: np.ndarray, season: int, H: int) -> Tuple[float, float, float]:
    """``(seasonal-naive, last-value, linear)`` MASE on the tail-``H`` horizon of ``x``,
    each scaled by the in-sample naive-1 diff. ``snaive < last`` ⇒ a learnable *recurring*
    pattern (seasonal-naive forecasts it); compare synth-vs-real **relatively**, never as an
    absolute threshold (fine sampling inflates MASE). ``snaive`` is ``nan`` when ``season<2``."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 * H + 2:
        return float("nan"), float("nan"), float("nan")
    ctx, y = x[:-H], x[-H:]
    scale = max(float(np.mean(np.abs(np.diff(ctx)))), 1e-3 * float(np.ptp(ctx)), 1e-6)
    last = float(np.mean(np.abs(ctx[-1] - y)) / scale)
    sn = float("nan")
    if season >= 2 and ctx.size >= season:
        reps = int(np.ceil(H / season))
        sn = float(np.mean(np.abs(np.tile(ctx[-season:], reps)[:H] - y)) / scale)
    lin = float("nan")
    k = min(ctx.size, 3 * H)
    t = np.arange(k, dtype=np.float64); tc = t - t.mean(); vt = float(np.dot(tc, tc))
    if vt > 1e-8:
        slope = float(np.dot(tc, ctx[-k:] - ctx[-k:].mean()) / vt)
        lin = float(np.mean(np.abs((ctx[-1] + slope * np.arange(1, H + 1)) - y)) / scale)
    return sn, last, lin


def _z(x: np.ndarray) -> np.ndarray:
    x = F._fill_nan(np.asarray(x, dtype=np.float64).ravel())
    return (x - x.mean()) / (x.std() + 1e-8)


def _real_items(config: str, n: int, *, local_dir: str = "", skip: int = 0):
    """First ``n`` real test items (after ``skip``) for ``config`` as ``(data[C,t], season)``."""
    from .gifteval_download import iter_eval_items
    out = []
    for i, ev in enumerate(iter_eval_items(local_dir, configs=[config],
                                           items_per_config=n + skip + 2, terms=("short",))):
        if i < skip:
            continue
        out.append((ev.data_tensor.detach().cpu().numpy(), int(ev.season_length or 0)))
        if len(out) >= n:
            break
    return out


# --- characterize -------------------------------------------------------------------------

def characterize(configs: List[str], *, n: int = 3, out_dir: str = ".",
                 local_dir: str = "") -> None:
    """Print the learnability split and write a multi-timescale, all-channel plot per config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        items = _real_items(cfg, n, local_dir=local_dir)
        if not items:
            print(f"[{cfg}] no items")
            continue
        arr, season = items[0]
        C, T = arr.shape
        H = int(min(max(4, season if season >= 2 else 12), max(4, T // 4)))
        print(f"\n==== {cfg} ====\n  channels={C} ctx_len={T} season(m)={season} H={H}")
        for c in range(min(C, 8)):
            sn, lv, ln = mase_split(arr[c], season, H)
            tag = "  snaive<last ✓LEARN" if (np.isfinite(sn) and sn < lv) else ""
            print(f"  ch{c}: snaive={sn:.3f} last={lv:.3f} lin={ln:.3f}{tag}")
        nch = min(C, 6)
        fig, ax = plt.subplots(nch, 3, figsize=(15, 2.0 * nch), squeeze=False)
        win_mid = season * 8 if season >= 2 else max(60, T // 6)
        win_sm = season * 2 if season >= 2 else 120
        for c in range(nch):
            x = arr[c]
            for col, (v, ttl) in enumerate([(x, "full"), (x[-min(T, win_mid):], f"~{win_mid}"),
                                            (x[-min(T, win_sm):], f"~{win_sm}")]):
                ax[c][col].plot(v, lw=0.6, color="tab:blue")
                ax[c][col].set_title(f"{cfg} ch{c} {ttl}", fontsize=7)
                ax[c][col].tick_params(labelsize=6)
        fig.tight_layout()
        p = Path(out_dir) / f"explore_{cfg.replace('/', '_')}.png"
        fig.savefig(p, dpi=105); plt.close(fig)
        print(f"  wrote {p}")


# --- real-vs-synth panels (the gate) ------------------------------------------------------

def panel(real_config: str, recipe: str, interval_min: int, out_path: str, *,
          n: int = 2, gen_len: int = 6000, local_dir: str = "", real_skip: int = 0,
          seed: int = 0) -> str:
    """Univariate real(blue)-vs-synth(orange) panel at 3 zoom levels (full / ~8 cycles /
    ~3 cycles), ``n`` z-scored exemplars per side. Synth is tailed to the real length."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reals = _real_items(real_config, n, local_dir=local_dir, skip=real_skip)
    m = reals[0][1] if reals else 0
    rlen = len(_z(reals[0][0][0])) if reals else gen_len
    syn = []
    for k in range(n):
        s = _z(R.gen_from_recipe(np.random.default_rng(seed + k), recipe, n=gen_len,
                                 interval_min=interval_min)[0])
        syn.append(s[-rlen:])
    reals_z = [_z(d[0]) for d, _ in reals]
    zooms = [("full", None), (f"~8 cyc ({8 * m})", 8 * m), (f"~3 cyc ({3 * m})", 3 * m)] \
        if m >= 2 else [("full", None), ("~600", 600), ("~200", 200)]
    fig, ax = plt.subplots(3, 2, figsize=(15, 7.5))
    fig.suptitle(f"{real_config}  vs  recipe='{recipe}'  (m={m}, Δ={interval_min}m)", fontsize=11)
    for row, (zlabel, zl) in enumerate(zooms):
        for col, (series, color, tag) in enumerate(((reals_z, "tab:blue", "REAL"),
                                                    (syn, "tab:orange", "SYNTH"))):
            for s in series:
                v = s if zl is None else s[-min(len(s), zl):]
                ax[row][col].plot(v, color=color, lw=0.7, alpha=0.85)
            ax[row][col].set_title(f"{tag} {real_config} [{zlabel}]", fontsize=8)
            ax[row][col].tick_params(labelsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
    return str(out_path)


def panel_multivar(real_config: str, recipe: str, interval_min: int, out_path: str, *,
                   gen_len: int = 4000, local_dir: str = "", seed: int = 1) -> str:
    """Multichannel panel: rows = channels, cols = REAL-8d / REAL-2d / SYNTH-8d / SYNTH-2d
    (for the composed multivariate recipes, e.g. ett/jena). Synth channel c is not meant to
    match real channel c exactly — judge whether the synth *set* reproduces the channel
    *kinds*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = _real_items(real_config, 1, local_dir=local_dir)[0]
    real, m = ev[0], ev[1] or 24
    syn = R.gen_from_recipe(np.random.default_rng(seed), recipe, n=gen_len,
                            interval_min=interval_min)
    C = real.shape[0]
    fig, ax = plt.subplots(C, 4, figsize=(17, 1.55 * C), squeeze=False)
    fig.suptitle(f"{real_config} (m={m}, Δ={interval_min}m) — REAL vs recipe '{recipe}', per channel",
                 fontsize=11)
    for c in range(C):
        rx, sx = _z(real[c]), _z(syn[min(c, syn.shape[0] - 1)])
        cells = [(rx[-m*8:], "tab:blue", "REAL 8d"), (rx[-m*2:], "tab:blue", "REAL 2d"),
                 (sx[-m*8:], "tab:orange", "SYNTH 8d"), (sx[-m*2:], "tab:orange", "SYNTH 2d")]
        for k, (v, col, t) in enumerate(cells):
            ax[c][k].plot(v, lw=0.6, color=col)
            ax[c][k].set_title(f"ch{c} {t}", fontsize=7); ax[c][k].tick_params(labelsize=5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100); plt.close(fig)
    return str(out_path)


def validate_corpus(corpus_dir: str, out_dir: str, *, k: int = 3, win: int = 1500,
                    per_montage: int = 12, local_dir: str = "", seed: int = 0) -> None:
    """End-to-end corpus validation: for each GIFT-Eval config, plot ``k`` random SYNTH series
    pulled from a materialized recipe corpus (filtered by ``kind`` = config name) against ``k``
    random REAL series, z-scored, side by side. Montages of ``per_montage`` configs each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from .shards import ShardReader

    rd = ShardReader(corpus_dir)
    by_kind: dict = defaultdict(list)
    for i in range(rd.n_series):
        by_kind[rd.meta(i)["kind"]].append(i)
    configs = sorted(by_kind)
    rng = np.random.default_rng(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def tail_z(x):
        z = _z(x); return z[-min(len(z), win):]

    for m0 in range(0, len(configs), per_montage):
        chunk = configs[m0:m0 + per_montage]
        fig, ax = plt.subplots(len(chunk), 2, figsize=(14, 1.5 * len(chunk)), squeeze=False)
        for row, cfg in enumerate(chunk):
            pick = rng.choice(by_kind[cfg], size=min(k, len(by_kind[cfg])), replace=False)
            syns = [tail_z(rd.read_array(int(g))[0]) for g in pick]
            try:
                reals = [tail_z(d[0]) for d, _ in _real_items(cfg, k, local_dir=local_dir)[:k]]
            except Exception:
                reals = []
            for col, (series, color, tag) in enumerate(((reals, "tab:blue", "REAL"),
                                                        (syns, "tab:orange", "SYNTH"))):
                for s in series:
                    ax[row][col].plot(s, lw=0.6, alpha=0.8, color=color)
                ax[row][col].set_title(f"{tag}  {cfg}  (n={len(series)})", fontsize=7)
                ax[row][col].tick_params(labelsize=5)
        fig.tight_layout()
        p = Path(out_dir) / f"validate_{m0 // per_montage:02d}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        print("wrote", p)


def main() -> None:  # pragma: no cover - manual entrypoint / produces artifacts
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("characterize", help="print learnability split + multi-scale plots")
    pc.add_argument("configs", nargs="+")
    pc.add_argument("--n", type=int, default=3)
    pc.add_argument("--out-dir", default="docs/tetris/sanity_experiments/synth_panels/explore")
    pc.add_argument("--local-dir", default="")

    pp = sub.add_parser("panel", help="univariate real-vs-synth panel")
    pp.add_argument("config"); pp.add_argument("recipe"); pp.add_argument("interval", type=int)
    pp.add_argument("--out", required=True)
    pp.add_argument("--n", type=int, default=2)
    pp.add_argument("--gen-len", type=int, default=6000)
    pp.add_argument("--real-skip", type=int, default=0)
    pp.add_argument("--local-dir", default="")

    pm = sub.add_parser("panel-mv", help="multichannel real-vs-synth panel")
    pm.add_argument("config"); pm.add_argument("recipe"); pm.add_argument("interval", type=int)
    pm.add_argument("--out", required=True)
    pm.add_argument("--gen-len", type=int, default=4000)
    pm.add_argument("--local-dir", default="")

    pv = sub.add_parser("validate-corpus", help="per-config REAL-vs-SYNTH montages from a recipe corpus")
    pv.add_argument("corpus"); pv.add_argument("--out-dir", required=True)
    pv.add_argument("--k", type=int, default=3)
    pv.add_argument("--win", type=int, default=1500)
    pv.add_argument("--per-montage", type=int, default=12)
    pv.add_argument("--local-dir", default="")

    a = ap.parse_args()
    if a.cmd == "characterize":
        characterize(a.configs, n=a.n, out_dir=a.out_dir, local_dir=a.local_dir)
    elif a.cmd == "panel":
        print("wrote", panel(a.config, a.recipe, a.interval, a.out, n=a.n,
                             gen_len=a.gen_len, real_skip=a.real_skip, local_dir=a.local_dir))
    elif a.cmd == "panel-mv":
        print("wrote", panel_multivar(a.config, a.recipe, a.interval, a.out,
                                      gen_len=a.gen_len, local_dir=a.local_dir))
    elif a.cmd == "validate-corpus":
        validate_corpus(a.corpus, a.out_dir, k=a.k, win=a.win,
                        per_montage=a.per_montage, local_dir=a.local_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
