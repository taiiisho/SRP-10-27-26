"""Error statistics, aggregated to TRIAL level first.

Frames of one trial are not independent measurements (they are 1/60 s apart
and share the same movement), so treating 10,000 frames as 10,000 samples
would make every difference look "significant". Instead:

1. For each trial, model, horizon and analysis window, summarise that
   trial's frame errors: mean, median, RMSE and 95th percentile.
2. Across trials, report the mean of the per-trial values, with a 95 %
   confidence interval from the bootstrap: resample whole trials with
   replacement thousands of times and see how much the statistic moves.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .predictions import WINDOWS


def trial_level(pred: pd.DataFrame, models: Sequence[str], windows: Sequence[str] = WINDOWS) -> pd.DataFrame:
    """One row per (window, model, horizon, trial) with that trial's error summary."""
    keys = ["trial_id", "participant_id", "speed_condition", "target_id", "split", "horizon_ms"]
    out = []
    for window in windows:
        sub = pred[pred[f"win_{window}"]]
        if sub.empty:
            continue
        grouped = sub.groupby(keys, sort=False)
        for model in models:
            col = f"{model}_error"
            g = grouped[col]
            frame = pd.DataFrame({
                "n_frames": g.count(),
                "mean": g.mean(),
                "median": g.median(),
                "rmse": np.sqrt(grouped[col].apply(lambda e: float(np.nanmean(e.to_numpy() ** 2)))),
                "p95": g.quantile(0.95),
            }).reset_index()
            frame.insert(0, "model", model)
            frame.insert(0, "window", window)
            out.append(frame[frame["n_frames"] > 0])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def bootstrap_ci(
    values: np.ndarray, stat: Callable[[np.ndarray], np.ndarray], n_boot: int, level: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval; ``stat`` works along axis 1 of the resampled matrix."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    reps = stat(values[idx])
    alpha = (1.0 - level) / 2.0
    return float(np.nanquantile(reps, alpha)), float(np.nanquantile(reps, 1.0 - alpha))


_STATS = {
    # name: (column of trial_level, statistic over trials, how to compute it on a matrix of resamples)
    "mean": ("mean", lambda v: np.mean(v), lambda m: m.mean(axis=1)),
    "median": ("mean", lambda v: np.median(v), lambda m: np.median(m, axis=1)),
    "rmse": ("rmse", lambda v: np.sqrt(np.mean(v**2)), lambda m: np.sqrt((m**2).mean(axis=1))),
    "p95": ("p95", lambda v: np.mean(v), lambda m: m.mean(axis=1)),
}


def model_level(
    trials: pd.DataFrame, split: str, n_boot: int, level: float, seed: int, models: Sequence[str] | None = None
) -> pd.DataFrame:
    """Statistics over trials, with bootstrap CIs, for every (window, model, horizon).

    mean   = mean over trials of each trial's mean error
    median = median over trials of each trial's mean error
    rmse   = root of the mean over trials of each trial's mean squared error
    p95    = mean over trials of each trial's 95th-percentile error
    """
    sel = trials[trials["split"] == split]
    if models is not None:
        sel = sel[sel["model"].isin(models)]
    rows = []
    for (window, model, horizon), g in sel.groupby(["window", "model", "horizon_ms"], sort=False):
        rng = np.random.default_rng([seed, int(horizon)])
        row = {"window": window, "model": model, "horizon_ms": horizon, "n_trials": len(g), "n_frames": int(g["n_frames"].sum())}
        for name, (col, point, matrix) in _STATS.items():
            v = g[col].to_numpy(float)
            row[name] = float(point(v))
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = bootstrap_ci(v, matrix, n_boot, level, rng)
        rows.append(row)
    return pd.DataFrame(rows)


def seed_variability(model_stats: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- sd over random seeds of each network type's mean error (single trained copies)."""
    seeds = model_stats[model_stats["model"].str.contains("@seed")].copy()
    if seeds.empty:
        return pd.DataFrame()
    seeds["base_model"] = seeds["model"].str.split("@").str[0]
    return (
        seeds.groupby(["window", "base_model", "horizon_ms"])["mean"]
        .agg(seed_mean="mean", seed_sd="std", n_seeds="count")
        .reset_index()
    )


def fallback_rates(pred: pd.DataFrame, models: Sequence[str], window: str) -> pd.DataFrame:
    """How often each physics model had to fall back to constant velocity, per horizon."""
    rows = []
    sub = pred[pred[f"win_{window}"]]
    for model in models:
        col = f"{model}_fallback"
        if col not in sub:
            continue
        for horizon, g in sub.groupby("horizon_ms"):
            reasons = g[col].fillna("").astype(str)
            rows.append({"model": model, "horizon_ms": horizon, "n": len(g),
                         "fallback_fraction": float((reasons != "").mean()),
                         "reasons": reasons[reasons != ""].value_counts().to_dict()})
    return pd.DataFrame(rows)
