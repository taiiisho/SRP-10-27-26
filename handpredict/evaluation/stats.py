"""Hypothesis tests: physics vs ML, per horizon, with trials as the unit.

* **Paired differences.** For every test trial, d = (ML trial mean error) -
  (physics trial mean error). Negative d means ML was better on that trial.
* **Wilcoxon signed-rank test** (per horizon): are the d's centred on zero?
  It uses only the signs and ranks of d, so it needs no normal distribution,
  which suits ~10-30 trials.
* **Effect size**: matched-pairs rank-biserial correlation r between -1 and
  +1 (r < 0: ML better on most / the larger differences; r > 0: physics
  better), plus the median difference in cm.
* **Holm correction**: testing 7 horizons gives 7 chances of a false alarm;
  Holm's method adjusts the p-values so the overall false-alarm rate stays
  at alpha.
* **Crossover**: the horizon where the mean difference changes sign from
  "physics better" to "ML better", with a bootstrap confidence interval.
* **Trend test**: does d get more negative as the horizon grows? (a slope per
  trial, then a Wilcoxon test of the slopes)
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .metrics import bootstrap_ci


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are ignored and kept as NaN)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (W+ - W-) / (W+ + W-), zeros dropped."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return float("nan")
    ranks = sps.rankdata(np.abs(d))
    w_plus, w_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


def wilcoxon_test(d: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test of d against 0: (statistic, p-value)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        return float("nan"), float("nan")
    res = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def min_possible_p(n: int) -> float:
    """Smallest two-sided p-value an exact Wilcoxon test can give with n pairs (all signs equal)."""
    return float(min(1.0, 2.0 / 2.0**n)) if n > 0 else float("nan")


def paired_matrix(trials: pd.DataFrame, window: str, split: str, physics: str, ml: str) -> pd.DataFrame:
    """Rows = trials, columns = horizons, values = ML - physics trial mean error."""
    sel = trials[(trials["window"] == window) & (trials["split"] == split) & trials["model"].isin([physics, ml])]
    wide = sel.pivot_table(index=["trial_id", "horizon_ms"], columns="model", values="mean")
    if physics not in wide or ml not in wide:
        return pd.DataFrame()
    diff = (wide[ml] - wide[physics]).dropna().rename("d").reset_index()
    return diff.pivot(index="trial_id", columns="horizon_ms", values="d").sort_index(axis=1)


def paired_comparison(
    trials: pd.DataFrame,
    window: str,
    split: str,
    physics: str,
    ml: str,
    n_boot: int,
    level: float,
    alpha: float,
    min_trials: int,
    seed: int,
) -> pd.DataFrame:
    """Per-horizon paired test of ML vs physics on whole trials."""
    dmat = paired_matrix(trials, window, split, physics, ml)
    sel = trials[(trials["window"] == window) & (trials["split"] == split)]
    rows = []
    for horizon in dmat.columns:
        d = dmat[horizon].dropna().to_numpy()
        n = len(d)
        rng = np.random.default_rng([seed, int(horizon), 7])
        phys = sel[(sel["model"] == physics) & (sel["horizon_ms"] == horizon) & sel["trial_id"].isin(dmat.index[dmat[horizon].notna()])]
        mlr = sel[(sel["model"] == ml) & (sel["horizon_ms"] == horizon) & sel["trial_id"].isin(dmat.index[dmat[horizon].notna()])]
        stat, p = wilcoxon_test(d) if n >= min_trials else (float("nan"), float("nan"))
        mean_lo, mean_hi = bootstrap_ci(d, lambda m: m.mean(axis=1), n_boot, level, rng)
        med_lo, med_hi = bootstrap_ci(d, lambda m: np.median(m, axis=1), n_boot, level, rng)
        rows.append({
            "window": window, "horizon_ms": horizon, "n_trials": n,
            "physics_model": physics, "ml_model": ml,
            "physics_mean": float(phys["mean"].mean()) if len(phys) else np.nan,
            "ml_mean": float(mlr["mean"].mean()) if len(mlr) else np.nan,
            "mean_diff": float(np.mean(d)) if n else np.nan, "mean_diff_ci_low": mean_lo, "mean_diff_ci_high": mean_hi,
            "median_diff": float(np.median(d)) if n else np.nan, "median_diff_ci_low": med_lo, "median_diff_ci_high": med_hi,
            "frac_trials_ml_better": float(np.mean(d < 0)) if n else np.nan,
            "wilcoxon_W": stat, "p_value": p, "rank_biserial": rank_biserial(d),
            "min_possible_p": min_possible_p(n),
            "tested": n >= min_trials,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_value"].to_numpy())
    tested = int(out["tested"].sum())
    out["min_possible_p_holm"] = np.minimum(1.0, out["min_possible_p"] * max(tested, 1))
    out["significant"] = out["p_holm"] < alpha
    out["better_model"] = np.where(~out["significant"], "no significant difference",
                                   np.where(out["mean_diff"] < 0, "ml", "physics"))
    return out


# -----------------------------------------------------------------------------
# Crossover
# -----------------------------------------------------------------------------
def classify_difference_curve(horizons: Sequence[float], diff: Sequence[float]) -> dict:
    """Describe a mean-difference curve D(h) = ML - physics (negative = ML better).

    Classes: "crossover" (physics better at short horizons, ML better at long
    ones; the hypothesis' pattern), "ml_better_at_all_horizons",
    "physics_better_at_all_horizons", "reverse_crossover", "multiple_crossings".
    Crossing points are found by straight-line interpolation between horizons.
    """
    h = np.asarray(horizons, dtype=float)
    d = np.asarray(diff, dtype=float)
    keep = np.isfinite(d) & (d != 0)  # an exact tie says nothing about the direction
    h, d = h[keep], d[keep]
    if len(d) == 0:
        return {"classification": "no_data", "crossings": [], "first_crossing_ms": None}
    if np.all(d < 0):
        return {"classification": "ml_better_at_all_horizons", "crossings": [], "first_crossing_ms": None}
    if np.all(d > 0):
        return {"classification": "physics_better_at_all_horizons", "crossings": [], "first_crossing_ms": None}
    crossings = []
    for k in range(len(d) - 1):
        if d[k] * d[k + 1] < 0:  # the sign changes between these two horizons
            frac = d[k] / (d[k] - d[k + 1])
            direction = "physics_to_ml" if d[k] > 0 else "ml_to_physics"
            crossings.append({"horizon_ms": float(h[k] + frac * (h[k + 1] - h[k])), "direction": direction})
    first = crossings[0]
    if len(crossings) == 1:
        cls = "crossover" if first["direction"] == "physics_to_ml" else "reverse_crossover"
    else:
        cls = "multiple_crossings"
    return {"classification": cls, "crossings": crossings, "first_crossing_ms": first["horizon_ms"]}


def crossover_analysis(dmat: pd.DataFrame, n_boot: int, level: float, seed: int) -> dict:
    """Observed crossover plus how often each pattern appears when trials are resampled."""
    horizons = np.asarray(dmat.columns, dtype=float)
    values = dmat.to_numpy(dtype=float)
    observed = classify_difference_curve(horizons, np.nanmean(values, axis=0) if len(values) else [])
    if len(values) < 2:
        return {"observed": observed, "bootstrap": None}
    rng = np.random.default_rng([seed, 99])
    classes: dict[str, int] = {}
    firsts = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        with np.errstate(all="ignore"):
            curve = np.nanmean(sample, axis=0)
        result = classify_difference_curve(horizons, curve)
        classes[result["classification"]] = classes.get(result["classification"], 0) + 1
        if result["classification"] == "crossover":
            firsts.append(result["first_crossing_ms"])
    alpha = (1 - level) / 2
    ci = [float(np.quantile(firsts, alpha)), float(np.quantile(firsts, 1 - alpha))] if len(firsts) >= 20 else None
    return {
        "observed": observed,
        "bootstrap": {
            "n_boot": n_boot,
            "pattern_fractions": {k: v / n_boot for k, v in sorted(classes.items(), key=lambda kv: -kv[1])},
            "crossover_ci_ms": ci,
            "crossover_median_ms": float(np.median(firsts)) if firsts else None,
        },
    }


def trend_test(dmat: pd.DataFrame, n_boot: int, level: float, seed: int, min_horizons: int = 3) -> dict:
    """Per-trial slope of d against horizon (cm per 100 ms) and a Wilcoxon test of the slopes."""
    slopes = []
    for _, row in dmat.iterrows():
        ok = row.notna().to_numpy()
        if ok.sum() < min_horizons:
            continue
        h = np.asarray(dmat.columns, dtype=float)[ok] / 100.0
        slopes.append(float(np.polyfit(h, row.to_numpy(dtype=float)[ok], 1)[0]))
    slopes_arr = np.asarray(slopes)
    if len(slopes_arr) == 0:
        return {"n_trials": 0}
    stat, p = wilcoxon_test(slopes_arr)
    lo, hi = bootstrap_ci(slopes_arr, lambda m: np.median(m, axis=1), n_boot, level, np.random.default_rng([seed, 5]))
    return {
        "n_trials": len(slopes_arr),
        "median_slope_cm_per_100ms": float(np.median(slopes_arr)),
        "median_slope_ci": [lo, hi],
        "frac_trials_negative_slope": float(np.mean(slopes_arr < 0)),
        "wilcoxon_W": stat,
        "p_value": p,
    }


def summarise_breakdown(trials: pd.DataFrame, window: str, split: str, by: str, models: Sequence[str]) -> pd.DataFrame:
    """Mean trial error per group (speed condition or target) - descriptive only, no tests."""
    sel = trials[(trials["window"] == window) & (trials["split"] == split) & trials["model"].isin(models)]
    return (
        sel.groupby([by, "model", "horizon_ms"])["mean"]
        .agg(mean_error="mean", sd_error="std", n_trials="count")
        .reset_index()
    )


def holm_reachable(n_trials: int, n_tests: int, alpha: float) -> bool:
    """Can ANY result with n trials reach significance after Holm correction over n_tests?"""
    return min_possible_p(n_trials) * n_tests < alpha


def as_records(df: pd.DataFrame) -> list[Mapping]:
    return df.replace({np.nan: None}).to_dict(orient="records")
