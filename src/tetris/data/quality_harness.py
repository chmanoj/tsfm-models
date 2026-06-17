"""Tier-1 synth-quality harness (H1) — *is the synthetic corpus good?*

Answers, **without a training run**, whether the synthetic families cover the real
GIFT-Eval test data distribution, on the shared feature space (:mod:`features`):

* **C2ST (Classifier Two-Sample Test)** — the headline objective metric. Train a
  classifier to tell synth from real test on the feature vectors and report ROC-AUC.
  The null is principled: **AUC ≈ 0.5 means the synth is statistically
  indistinguishable** from test on these features; AUC ≈ 1.0 means trivially
  separable. Unlike an arbitrary MMD/KS threshold this has a calibrated target, and
  the per-feature KS table is the punch-list of *what still gives the synth away*.
* **Per-feature KS** distance + **RBF-MMD** as cross-checks / diagnostics.
* A **pure-noise control** anchors the scale: a good targeted family's C2ST AUC must
  be far below the noise control's (which should sit near 1.0).

Verdict (relative, null-anchored): ``AUC(targeted) < AUC(general) ≪ AUC(noise)`` and
both real families' AUC well below the control. Pure numpy (no sklearn/scipy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import features as F


# --- metric primitives -------------------------------------------------------

def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann–Whitney U statistic (labels in {0,1})."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _logreg_scores(Xtr, ytr, Xte, *, iters=300, lr=0.5, l2=1e-3) -> np.ndarray:
    """Tiny L2 logistic regression (batch GD); returns scores for ``Xte``."""
    n, d = Xtr.shape
    w = np.zeros(d); b = 0.0
    # np.errstate guards a known macOS NumPy+Accelerate spurious matmul warning; the
    # sigmoid is clipped so the arithmetic is genuinely safe.
    with np.errstate(all="ignore"):
        for _ in range(iters):
            z = Xtr @ w + b
            pr = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = pr - ytr
            w -= lr * (Xtr.T @ g / n + l2 * w)
            b -= lr * float(g.mean())
        return Xte @ w + b


def _augment(X: np.ndarray) -> np.ndarray:
    """Append squared terms so a linear classifier can pick up axis-wise nonlinearity."""
    return np.concatenate([X, X ** 2], axis=1)


def c2st_auc(A: np.ndarray, B: np.ndarray, *, n_folds: int = 5, seed: int = 0) -> float:
    """Classifier two-sample test AUC between feature sets ``A`` (label 0) and ``B``
    (label 1) via k-fold cross-validated logistic regression. ~0.5 ⇒ indistinguishable."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if len(A) < n_folds or len(B) < n_folds:
        return 0.5
    X = _augment(np.concatenate([A, B], axis=0))
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sd
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, n_folds)
    scores = np.empty(len(X));
    for k in range(n_folds):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        scores[te] = _logreg_scores(X[tr], y[tr], X[te])
    return auc_score(scores, y)


def knn_c2st_auc(A: np.ndarray, B: np.ndarray, *, k: int = 15, seed: int = 0) -> float:
    """A stronger (nonlinear) C2ST: leave-fold-out k-NN. Each point's score is the
    fraction of its k nearest neighbours (among the *other* points) labelled 1. Picks
    up separability a linear classifier misses, so it is the more honest (usually
    higher) AUC — we report it to avoid flattering ourselves."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if len(A) < k + 1 or len(B) < k + 1:
        return 0.5
    X = np.concatenate([A, B], axis=0)
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sd
    with np.errstate(all="ignore"):  # Accelerate matmul warning (see _logreg_scores)
        d2 = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * X @ X.T
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    scores = y[nn].mean(axis=1)
    return auc_score(scores, y)


def ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic max|F_a − F_b| (numpy)."""
    a = np.sort(np.asarray(a, dtype=np.float64)); b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return 0.0
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def per_feature_ks(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    return {name: ks_2samp(A[:, j], B[:, j]) for j, name in enumerate(F.FEATURE_NAMES)}


def rbf_mmd(A: np.ndarray, B: np.ndarray, *, gamma: Optional[float] = None) -> float:
    """Unbiased RBF-kernel MMD² between feature sets (standardized; median-heuristic γ)."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if len(A) < 2 or len(B) < 2:
        return 0.0
    pooled = np.concatenate([A, B], axis=0)
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-8
    A = (A - mu) / sd; B = (B - mu) / sd

    def sq(X, Y):
        with np.errstate(all="ignore"):  # see _logreg_scores (Accelerate matmul warning)
            return (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T

    if gamma is None:
        d2 = sq(pooled[:200], pooled[:200])  # median heuristic on a subsample
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1.0 / (med + 1e-8)
    Kaa, Kbb, Kab = np.exp(-gamma * sq(A, A)), np.exp(-gamma * sq(B, B)), np.exp(-gamma * sq(A, B))
    na, nb = len(A), len(B)
    term_a = (Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
    term_b = (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
    return float(term_a + term_b - 2 * Kab.mean())


# --- learnability: is the synth forecastable by simple baselines? ------------

def _baseline_mase(x: np.ndarray, season: int) -> float:
    """Best-of {seasonal-naive, last-value, linear} forecast error on the tail horizon,
    scaled by the naive-1 in-sample diff (MASE-like). Low ⇒ the series has *predictable
    structure*; ~1+ ⇒ a simple model can't beat naive ⇒ effectively unpredictable noise."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    H = int(min(max(4, season if season >= 2 else 12), max(4, n // 4)))
    if n < 2 * H + 2:
        return np.nan
    ctx, y = x[:-H], x[-H:]
    # robust scale: in-sample naive-1 diff, floored by a fraction of the series range so
    # a near-flat series with rare spikes can't produce a divide-by-tiny blow-up.
    scale = max(float(np.mean(np.abs(np.diff(ctx)))),
                1e-3 * float(np.ptp(ctx)), 1e-6)
    preds = [np.full(H, ctx[-1])]                                  # last-value
    if season >= 2 and ctx.size >= season:                        # seasonal-naive
        reps = int(np.ceil(H / season))
        preds.append(np.tile(ctx[-season:], reps)[:H])
    k = min(ctx.size, 3 * H)                                       # linear extrapolation
    t = np.arange(k, dtype=np.float64); tc = t - t.mean()
    vt = float(np.dot(tc, tc))
    if vt > 1e-8:
        slope = float(np.dot(tc, ctx[-k:] - ctx[-k:].mean()) / vt)
        preds.append(ctx[-1] + slope * np.arange(1, H + 1))
    return float(min(np.mean(np.abs(p - y)) / scale for p in preds))


def series_learnability(series, seasons) -> float:
    """**Median** best-baseline MASE over a list of series (lower = more forecastable).
    Median (not mean) so a few hard/pathological series don't dominate the aggregate."""
    vals = [_baseline_mase(x, int(s)) for x, s in zip(series, seasons)]
    vals = [min(v, 20.0) for v in vals if np.isfinite(v)]          # cap per-series outliers
    return float(np.median(vals)) if vals else float("nan")


# --- predictability floor (noise-robust families) ----------------------------

def predictability_floor(rng, n_samples: int = 200, ctx: int = 256, horizon: int = 32) -> float:
    """Mean fraction of the clean-horizon scale that is *irreducible noise*.

    For each noise-robust process we have the clean conditional mean and can resimulate
    a noisy realization of the same horizon; the ratio of their RMS gap to the clean
    horizon scale is the floor a perfect forecaster still pays. Reported so a low C2ST
    isn't mistaken for "learnable to zero error"."""
    from . import synthetic_v2 as V
    fracs = []
    for _ in range(n_samples):
        s, kind = V.gen_noise_robust(rng, ctx, horizon)
        clean = s[ctx:]
        # resimulate one noisy realization continuation around the clean mean
        resid_scale = float(np.std(np.diff(s[:ctx]))) + 1e-8
        noisy = clean + rng.normal(0, resid_scale, horizon)
        gap = float(np.sqrt(np.mean((noisy - clean) ** 2)))
        scale = float(np.std(clean)) + resid_scale + 1e-8
        fracs.append(gap / scale)
    return float(np.mean(fracs))


# --- orchestration -----------------------------------------------------------

@dataclass
class QualityResult:
    real_n: int
    families: Dict[str, dict] = field(default_factory=dict)  # name -> {n, c2st_*, mmd, ks, learnability}
    predictability_floor: Optional[float] = None
    real_learnability: Optional[float] = None
    verdict: str = ""

    def to_dict(self) -> dict:
        return {"real_n": self.real_n, "families": self.families,
                "predictability_floor": self.predictability_floor,
                "real_learnability": self.real_learnability, "verdict": self.verdict}


def evaluate(real_feats: np.ndarray, named_synth: Dict[str, np.ndarray], *,
             seed: int = 0, floor: Optional[float] = None,
             learnability: Optional[Dict[str, float]] = None,
             real_learnability: Optional[float] = None) -> QualityResult:
    """Compare each named synth feature set against the real-test feature set."""
    res = QualityResult(real_n=int(len(real_feats)), predictability_floor=floor,
                        real_learnability=real_learnability)
    learnability = learnability or {}
    dyn = list(F.DYNAMICS_IDX)
    real_dyn = real_feats[:, dyn]
    for name, feats in named_synth.items():
        feats = np.asarray(feats, dtype=np.float64)
        res.families[name] = {
            "learnability": learnability.get(name),
            "n": int(len(feats)),
            # headline = the *dynamics* C2ST (length/scale excluded so it can't be
            # gamed by superficial matching); full-feature + kNN reported alongside.
            "c2st_dynamics": c2st_auc(feats[:, dyn], real_dyn, seed=seed),
            "c2st_full": c2st_auc(feats, real_feats, seed=seed),
            "c2st_knn_dynamics": knn_c2st_auc(feats[:, dyn], real_dyn, seed=seed),
            "mmd": rbf_mmd(feats[:, dyn], real_dyn),
            "ks": per_feature_ks(feats, real_feats),
        }
    # Verdict on the *dynamics* C2ST. Two relative checks (robust to absolute level):
    # (1) targeted closer to the 0.5 null than general; (2) the pure-noise control is
    # the most separable family (the C2ST has discriminative power). Plus an honest
    # absolute caveat — how far the best real family still is from indistinguishable.
    aucs = {n: v["c2st_dynamics"] for n, v in res.families.items()}
    parts = [f"{n}:dyn={aucs[n]:.3f}/knn={res.families[n]['c2st_knn_dynamics']:.3f}"
             for n in sorted(aucs, key=aucs.get)]
    real = {n: a for n, a in aucs.items() if n != "noise"}
    rel_ok = True
    if "targeted" in aucs and "general" in aucs:
        rel_ok = rel_ok and abs(aucs["targeted"] - 0.5) <= abs(aucs["general"] - 0.5) + 1e-9
    if "noise" in aucs and real:
        rel_ok = rel_ok and aucs["noise"] >= max(real.values())
    best = min(real.values()) if real else min(aucs.values())
    absolute = ("indistinguishable" if best < 0.6 else
                "good overlap" if best < 0.8 else
                "still separable — capture more real dynamics (not length/scale)")
    res.verdict = (("PASS(relative) — " if rel_ok else "REVIEW — ")
                   + " | ".join(parts)
                   + f" || best targeted/general dynamics-AUC={best:.3f} → {absolute}"
                   + " (0.5 = indistinguishable; dyn excludes length/scale)")
    return res


def format_report(res: QualityResult) -> str:
    lines = ["# Tier-1 synth-quality report", "",
             f"Real GIFT-Eval test feature rows: **{res.real_n}**", "",
             f"**Verdict:** {res.verdict}", ""]
    if res.predictability_floor is not None:
        lines += [f"Noise-robust predictability floor (irreducible RMS / scale): "
                  f"**{res.predictability_floor:.3f}**", ""]
    if res.real_learnability is not None:
        lines += ["## Learnability (best-baseline MASE on the tail horizon — lower = more "
                  "forecastable; should be near real, not ≫)", "",
                  f"real test: **{res.real_learnability:.3f}**", "",
                  "| family | learnability |", "|---|---|"]
        for name, v in sorted(res.families.items(),
                              key=lambda kv: (kv[1].get("learnability") is None,
                                              kv[1].get("learnability") or 0)):
            lv = v.get("learnability")
            lines.append(f"| {name} | {lv:.3f} |" if lv is not None else f"| {name} | – |")
        lines.append("")
    lines += ["## C2ST / MMD (lower AUC = closer to test; 0.5 = indistinguishable). "
              "Headline = **dynamics** AUC (length/scale excluded — not gameable).", "",
              "| family | n | C2ST dyn | C2ST dyn (kNN) | C2ST full | RBF-MMD(dyn) |",
              "|---|---|---|---|---|---|"]
    for name, v in sorted(res.families.items(), key=lambda kv: kv[1]["c2st_dynamics"]):
        lines.append(f"| {name} | {v['n']} | {v['c2st_dynamics']:.3f} | "
                     f"{v['c2st_knn_dynamics']:.3f} | {v['c2st_full']:.3f} | {v['mmd']:.4f} |")
    lines += ["", "## Per-feature KS distance (the punch-list: what still separates)", "",
              "| feature | " + " | ".join(res.families) + " |",
              "|---|" + "---|" * len(res.families)]
    for feat in F.FEATURE_NAMES:
        row = " | ".join(f"{res.families[n]['ks'][feat]:.2f}" for n in res.families)
        lines.append(f"| {feat} | {row} |")
    return "\n".join(lines) + "\n"


def main() -> None:  # pragma: no cover - manual entrypoint
    import argparse
    from pathlib import Path
    from .test_profile import TestProfile
    from .gifteval_download import iter_eval_items
    from . import synthetic_v2 as V
    from . import synthetic_targeted as TGT

    ap = argparse.ArgumentParser(description="Tier-1 synth-quality harness")
    ap.add_argument("--profile", required=True, help="committed TestProfile JSON")
    ap.add_argument("--local-dir", default="", help="GIFT-Eval test root (else $GIFT_EVAL)")
    ap.add_argument("--n", type=int, default=600, help="synth series per family")
    ap.add_argument("--items-per-config", type=int, default=40)
    ap.add_argument("--report", default="", help="write markdown report here")
    ap.add_argument("--json", default="", help="write JSON results here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    profile = TestProfile.load(args.profile)
    # real-test features (season-aware; aggregate per-series stats, never values) + raw
    # series/seasons for the learnability gate.
    real, real_series, real_seasons = [], [], []
    for ev in iter_eval_items(args.local_dir, items_per_config=args.items_per_config,
                              terms=("short", "medium", "long")):
        data = ev.data_tensor.detach().cpu().numpy()
        season = int(ev.season_length) if ev.season_length else 0
        real.append(F.channel_features(data, season=season))
        real_series.append(data[0]); real_seasons.append(season)
    real = np.concatenate(real, axis=0)

    rng = np.random.default_rng(args.seed)
    feats = {"general": [], "targeted": [], "noise": []}
    series = {"general": [], "targeted": [], "noise": []}
    seasons = {"general": [], "targeted": [], "noise": []}
    for i in range(args.n):
        fam, p = V.general_picker(None)
        f = fam[int(rng.choice(len(fam), p=p))]
        gd = V.gen_general_series(np.random.default_rng((args.seed, 1, i)), int(rng.integers(200, 1200)), f)
        feats["general"].append(F.channel_features(np.atleast_2d(gd[0]))); series["general"].append(np.atleast_2d(gd[0])[0]); seasons["general"].append(gd[3] if gd[3] > 0 else 0)
        td, _nf, _nt, tm, _g = TGT.gen_targeted(np.random.default_rng((args.seed, 2, i)), profile)
        feats["targeted"].append(F.channel_features(np.atleast_2d(td), season=max(0, tm))); series["targeted"].append(np.atleast_2d(td)[0]); seasons["targeted"].append(tm if tm > 0 else 0)
        nz = rng.normal(0, 1, int(rng.integers(200, 1200)))
        feats["noise"].append(F.series_features(nz)[None, :]); series["noise"].append(nz); seasons["noise"].append(0)
    named = {k: np.concatenate(v) for k, v in feats.items()}
    learn = {k: series_learnability(series[k], seasons[k]) for k in series}
    res = evaluate(real, named, seed=args.seed,
                   floor=predictability_floor(np.random.default_rng(args.seed)),
                   learnability=learn,
                   real_learnability=series_learnability(real_series, real_seasons))
    report = format_report(res)
    print(report)
    if args.report:
        from pathlib import Path
        Path(args.report).write_text(report)
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
