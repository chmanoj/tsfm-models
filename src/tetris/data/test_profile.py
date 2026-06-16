"""TestProfile (H1) — the empirical GIFT-Eval-test *data distribution*, as aggregate
feature statistics, used to condition the targeted synthetic family.

The maintainer's requirement: matching freq/season/horizon/C *metadata* is not
enough — the targeted synth must look like the test data in **value space**. So we
fit, per frequency, the empirical distribution of the canonical feature battery
(:mod:`tetris.data.features`) over the real test series, plus the structural
marginals (season ``m``, horizon, channel count). The targeted generator then
samples a frequency group and rejection-samples candidates whose feature vector
lands inside that group's feature bands.

**No leakage:** a profile stores only *aggregate statistics* (per-feature quantiles
and marginal value→weight maps), never raw test values, and matching is at the
group level, never per-series. The model never sees the profile; it only shapes the
synthetic data distribution. This is the "match marginals, never values" rule (D13)
extended from metadata to the full feature distribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import features as F

PROFILE_VERSION = 1
_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class FitRecord:
    """One real series' contribution to the profile."""
    feats: np.ndarray   # [C, N_FEATURES] per-channel feature rows
    freq: str
    season: int
    horizon: int
    n_channels: int


def _marginal(values: List[int]) -> Dict[str, list]:
    """Empirical value→count marginal as parallel ``values``/``weights`` lists."""
    if not values:
        return {"values": [], "weights": []}
    uniq, cnt = np.unique(np.asarray(values), return_counts=True)
    return {"values": [int(v) for v in uniq], "weights": [int(c) for c in cnt]}


class TestProfile:
    """Per-frequency feature-distribution + structural-marginal profile."""

    __test__ = False  # not a pytest test class (name starts with 'Test')

    def __init__(self, groups: Dict[str, dict], feature_names: List[str]) -> None:
        self.groups = groups
        self.feature_names = feature_names

    # --- fitting -------------------------------------------------------------

    @classmethod
    def fit(cls, records: Iterable[FitRecord]) -> "TestProfile":
        by_freq: Dict[str, dict] = {}
        for r in records:
            g = by_freq.setdefault(r.freq, {"feats": [], "season": [], "horizon": [],
                                            "C": [], "n": 0})
            g["feats"].append(np.atleast_2d(r.feats))
            if r.season > 0:
                g["season"].append(int(r.season))
            if r.horizon > 0:
                g["horizon"].append(int(r.horizon))
            g["C"].append(int(r.n_channels))
            g["n"] += 1
        groups: Dict[str, dict] = {}
        for freq, g in by_freq.items():
            feats = np.concatenate(g["feats"], axis=0)  # [sum_C, F]
            qs = np.quantile(feats, _QUANTILES, axis=0)  # [5, F]
            groups[freq] = {
                "weight": int(g["n"]),
                "n_channels_total": int(feats.shape[0]),
                "feature_quantiles": {
                    name: [float(qs[i, j]) for i in range(len(_QUANTILES))]
                    for j, name in enumerate(F.FEATURE_NAMES)
                },
                "feature_mean": [float(v) for v in feats.mean(axis=0)],
                "feature_std": [float(v) for v in feats.std(axis=0)],
                "season": _marginal(g["season"]),
                "horizon": _marginal(g["horizon"]),
                "C": _marginal(g["C"]),
            }
        return cls(groups, list(F.FEATURE_NAMES))

    # --- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {"version": PROFILE_VERSION, "feature_names": self.feature_names,
                "n_series": int(sum(g["weight"] for g in self.groups.values())),
                "groups": self.groups}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "TestProfile":
        d = json.loads(Path(path).read_text())
        if d.get("version") != PROFILE_VERSION:
            raise ValueError(f"unsupported profile version {d.get('version')!r}")
        return cls(d["groups"], d["feature_names"])

    # --- sampling helpers (for the targeted generator) -----------------------

    def sample_group(self, rng) -> str:
        keys = list(self.groups.keys())
        w = np.array([self.groups[k]["weight"] for k in keys], dtype=np.float64)
        return keys[int(rng.choice(len(keys), p=w / w.sum()))]

    def feature_bands(self, group: str, *, lo: int = 0, hi: int = 4
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-feature acceptance band (default q05..q95) for rejection sampling."""
        q = self.groups[group]["feature_quantiles"]
        names = F.FEATURE_NAMES
        return (np.array([q[n][lo] for n in names], dtype=np.float64),
                np.array([q[n][hi] for n in names], dtype=np.float64))

    def feature_center(self, group: str) -> np.ndarray:
        return np.array(self.groups[group]["feature_mean"], dtype=np.float64)

    def feature_scale(self, group: str) -> np.ndarray:
        return np.array(self.groups[group]["feature_std"], dtype=np.float64) + 1e-6

    def _sample_marginal(self, group: str, key: str, rng, default: int) -> int:
        m = self.groups[group][key]
        if not m["values"]:
            return default
        w = np.array(m["weights"], dtype=np.float64)
        return int(m["values"][int(rng.choice(len(m["values"]), p=w / w.sum()))])

    def sample_season(self, group: str, rng) -> int:
        return self._sample_marginal(group, "season", rng, default=1)

    def sample_horizon(self, group: str, rng) -> int:
        return self._sample_marginal(group, "horizon", rng, default=24)

    def sample_n_channels(self, group: str, rng) -> int:
        return max(1, self._sample_marginal(group, "C", rng, default=1))


# --- fitting from the real GIFT-Eval test set (Mac/WSL; lazy gluonts) --------

_TERMS = ("short", "medium", "long")


def _config_name(config_id: str) -> str:
    """Strip the trailing ``/{term}`` from a ``"{name}/{term}"`` config id."""
    head, _, tail = config_id.rpartition("/")
    return head if (head and tail in _TERMS) else config_id


def fit_from_gifteval(*, local_dir: str = "", terms=_TERMS, items_per_config: int = 50,
                      configs=None) -> TestProfile:  # pragma: no cover - requires real data
    """Fit a :class:`TestProfile` from the downloaded GIFT-Eval **test** split.

    Reads the held-out *contexts* (never the horizon values) via ``iter_eval_items``
    and summarizes their feature distribution **per config** (term stripped, so each
    of the ~97 GIFT-Eval configs is one group). Lazy/network — needs the ``gift_eval``
    extras + a populated tree (``~/Projects/gifteval`` or ``$GIFT_EVAL``). Stores only
    aggregate statistics, never raw series (no leakage)."""
    from .gifteval_download import iter_eval_items

    records: List[FitRecord] = []
    for ev in iter_eval_items(local_dir, configs=configs, terms=tuple(terms),
                              items_per_config=items_per_config):
        data = ev.data_tensor.detach().cpu().numpy()
        season = int(ev.season_length) if ev.season_length else 0
        records.append(FitRecord(
            feats=F.channel_features(data, season=season),
            freq=_config_name(ev.config_id),
            season=season,
            horizon=int(ev.y_true.shape[0]) if ev.y_true is not None else 0,
            n_channels=data.shape[0]))
    if not records:
        raise ValueError("no GIFT-Eval test series read; check the download / local_dir")
    return TestProfile.fit(records)


def main() -> None:  # pragma: no cover - manual entrypoint
    import argparse
    ap = argparse.ArgumentParser(description="Fit a GIFT-Eval TestProfile (aggregate, no leakage)")
    ap.add_argument("--out", required=True, help="output profile JSON path")
    ap.add_argument("--local-dir", default="", help="GIFT-Eval test root (else $GIFT_EVAL)")
    ap.add_argument("--items-per-config", type=int, default=50)
    args = ap.parse_args()
    prof = fit_from_gifteval(local_dir=args.local_dir, items_per_config=args.items_per_config)
    prof.save(args.out)
    print(f"fit profile: {len(prof.groups)} groups, "
          f"{sum(g['weight'] for g in prof.groups.values())} series -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
