"""Figures for the results (PNG) plus the numbers behind each one (CSV).

How to read them is explained in the README ("Interpreting the plots").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..plotting import MARKERS, OKABE_ITO, model_colour, new_figure, save_figure

log = logging.getLogger(__name__)


def _label(model: str, headline: Sequence[str]) -> str:
    return f"{model} (headline)" if model in headline else model


def plot_error_vs_horizon(stats: pd.DataFrame, window: str, models: Sequence[str], headline: Sequence[str],
                          unit: str, out_dir: Path, split: str = "test", name: str | None = None, log_y: bool = False) -> None:
    """Mean error (95% CI) against horizon; ``log_y`` keeps exploding baselines from hiding the rest."""
    name = name or f"error_vs_horizon_{window}"
    sel = stats[(stats["window"] == window) & stats["model"].isin(models)].sort_values("horizon_ms")
    sel.to_csv(out_dir / f"{name}.csv", index=False)
    fig, ax = new_figure(8.0, 5.0)
    for i, model in enumerate(models):
        g = sel[sel["model"] == model]
        if g.empty:
            continue
        lw, ms = (2.6, 7) if model in headline else (1.2, 4)
        yerr = np.vstack([g["mean"] - g["mean_ci_low"], g["mean_ci_high"] - g["mean"]])
        ax.errorbar(g["horizon_ms"] + (i - len(models) / 2) * 2.0, g["mean"], yerr=yerr, color=model_colour(model, i),
                    marker=MARKERS[i % len(MARKERS)], lw=lw, ms=ms, capsize=3, label=_label(model, headline))
    ax.set_xlabel("prediction horizon (ms)")
    ax.set_ylabel(f"mean prediction error ({unit}), 95% CI" + (", log scale" if log_y else ""))
    ax.set_title(f"Error vs horizon - {split} trials, window: {window}")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, out_dir / f"{name}.png")


def plot_difference_vs_horizon(paired: pd.DataFrame, crossover: Mapping, window: str, unit: str, out_dir: Path) -> None:
    if paired.empty:
        return
    p = paired.sort_values("horizon_ms")
    p.to_csv(out_dir / f"difference_vs_horizon_{window}.csv", index=False)
    fig, ax = new_figure(8.0, 4.8)
    ax.axhline(0, color="black", lw=1)
    ax.fill_between(p["horizon_ms"], p["mean_diff_ci_low"], p["mean_diff_ci_high"], color=OKABE_ITO["sky_blue"], alpha=0.35,
                    label="95% CI (bootstrap over trials)")
    ax.plot(p["horizon_ms"], p["mean_diff"], "o-", color=OKABE_ITO["blue"], lw=2, label="mean difference")
    sig = p[p["significant"]]
    ax.plot(sig["horizon_ms"], sig["mean_diff"], "*", ms=14, color=OKABE_ITO["vermillion"], label="significant (Holm)")
    obs = crossover.get("observed", {})
    if obs.get("classification") == "crossover":
        ax.axvline(obs["first_crossing_ms"], color=OKABE_ITO["green"], ls="--", label=f"crossover {obs['first_crossing_ms']:.0f} ms")
        ci = (crossover.get("bootstrap") or {}).get("crossover_ci_ms")
        if ci:
            ax.axvspan(ci[0], ci[1], color=OKABE_ITO["green"], alpha=0.12, label="crossover 95% CI")
    ax.set_xlabel("prediction horizon (ms)")
    ax.set_ylabel(f"ML error - physics error ({unit})")
    ax.set_title(f"Below 0 = ML better, above 0 = physics better (test trials, {window})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, out_dir / f"difference_vs_horizon_{window}.png")


def plot_paired_differences(dmat: pd.DataFrame, window: str, unit: str, out_dir: Path) -> None:
    if dmat.empty:
        return
    long = dmat.reset_index().melt(id_vars="trial_id", var_name="horizon_ms", value_name="ml_minus_physics").dropna()
    long.to_csv(out_dir / f"paired_differences_{window}.csv", index=False)
    fig, ax = new_figure(8.0, 4.8)
    rng = np.random.default_rng(0)
    horizons = sorted(long["horizon_ms"].unique())
    for k, h in enumerate(horizons):
        d = long.loc[long["horizon_ms"] == h, "ml_minus_physics"].to_numpy()
        ax.scatter(np.full(len(d), k) + rng.uniform(-0.15, 0.15, len(d)), d, s=18, color=OKABE_ITO["blue"], alpha=0.7)
        ax.hlines(np.median(d), k - 0.3, k + 0.3, color=OKABE_ITO["vermillion"], lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h:g}" for h in horizons])
    ax.set_xlabel("prediction horizon (ms)")
    ax.set_ylabel(f"ML - physics, per test trial ({unit})")
    ax.set_title("Each dot is one test trial; red line = median; below 0 = ML better", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, out_dir / f"paired_differences_{window}.png")


def plot_example_trajectories(pred: pd.DataFrame, trial_ids: Sequence[str], horizons: Sequence[float],
                              physics: str, ml: str, unit: str, out_dir: Path) -> None:
    if not trial_ids:
        return
    fig, axes = new_figure(4.0 * len(horizons), 3.8 * len(trial_ids), nrows=len(trial_ids), ncols=len(horizons), squeeze=False)
    rows = []
    for r, tid in enumerate(trial_ids):
        t = pred[pred["trial_id"] == tid]
        path = t[t["horizon_ms"] == t["horizon_ms"].min()].sort_values("frame_index")
        for c, h in enumerate(horizons):
            ax = axes[r][c]
            g = t[(t["horizon_ms"] == h) & t["in_movement"] & t["gt_valid"]]
            ax.plot(path["x"], path["y"], "-", color="#7F7F7F", lw=1.5, label="measured path")
            ax.plot(g["actual_x"], g["actual_y"], ".", ms=4, color=OKABE_ITO["black"], label=f"actual at t+{h:g} ms")
            ax.plot(g[f"{physics}_pred_x"], g[f"{physics}_pred_y"], "o", ms=3, mfc="none",
                    color=model_colour(physics), label=f"{physics}")
            ax.plot(g[f"{ml}_pred_x"], g[f"{ml}_pred_y"], "x", ms=4, color=model_colour(ml), label=f"{ml}")
            ax.set_aspect("equal", adjustable="datalim")
            ax.invert_yaxis()  # image coordinates: y grows downwards, as the camera sees it
            ax.set_title(f"{tid.split('/')[-1]} - horizon {h:g} ms", fontsize=9)
            ax.set_xlabel(f"x ({unit})")
            ax.set_ylabel(f"y ({unit})")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")
            rows.append(g.assign(panel_horizon=h)[["trial_id", "frame_index", "horizon_ms", "x", "y", "actual_x", "actual_y",
                                                    f"{physics}_pred_x", f"{physics}_pred_y", f"{ml}_pred_x", f"{ml}_pred_y"]])
    if rows:
        pd.concat(rows).to_csv(out_dir / "example_trajectories.csv", index=False)
    save_figure(fig, out_dir / "example_trajectories.png")


def plot_error_vs_speed(pred: pd.DataFrame, window: str, horizons: Sequence[float], models: Sequence[str],
                        unit: str, out_dir: Path, n_bins: int = 8) -> None:
    sub = pred[pred[f"win_{window}"] & (pred["split"] == "test")]
    if sub.empty:
        return
    fig, axes = new_figure(4.5 * len(horizons), 4.0, ncols=len(horizons), squeeze=False)
    rows = []
    for c, h in enumerate(horizons):
        g = sub[sub["horizon_ms"] == h]
        if len(g) < n_bins:
            continue
        bins = np.unique(np.quantile(g["speed"], np.linspace(0, 1, n_bins + 1)))
        centres = 0.5 * (bins[1:] + bins[:-1])
        which = np.clip(np.digitize(g["speed"], bins[1:-1]), 0, len(centres) - 1)
        ax = axes[0][c]
        for i, model in enumerate(models):
            means = [g.loc[which == b, f"{model}_error"].mean() for b in range(len(centres))]
            ax.plot(centres, means, marker=MARKERS[i % len(MARKERS)], color=model_colour(model, i), label=model)
            rows += [{"horizon_ms": h, "model": model, "speed_bin_centre": sc, "mean_error": m} for sc, m in zip(centres, means)]
        ax.set_xlabel(f"hand speed at prediction time ({unit}/s)")
        ax.set_ylabel(f"mean error ({unit})")
        ax.set_title(f"horizon {h:g} ms (test frames, {window})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    pd.DataFrame(rows).to_csv(out_dir / "error_vs_speed.csv", index=False)
    save_figure(fig, out_dir / "error_vs_speed.png")


def plot_learning_curves(models_dir: Path, summary: Mapping, out_dir: Path) -> None:
    for name, card in summary.get("models", {}).items():
        if card["kind"] == "gp":
            continue
        final_path = models_dir / name / "final_histories.csv"
        cv_path = models_dir / name / "cv_histories.csv"
        if not final_path.exists():
            continue
        final = pd.read_csv(final_path)
        cv = pd.read_csv(cv_path) if cv_path.exists() else pd.DataFrame()
        fig, (ax1, ax2) = new_figure(10.0, 4.0, ncols=2)
        if not cv.empty:
            sel = cv[cv["config"] == card.get("selected_config", 0)]
            for fold, g in sel.groupby("fold"):
                ax1.plot(g["epoch"], g["train_loss"], color=OKABE_ITO["blue"], alpha=0.6, lw=1, label="train" if fold == 0 else None)
                ax1.plot(g["epoch"], g["val_loss"], color=OKABE_ITO["vermillion"], alpha=0.8, lw=1, label="validation fold" if fold == 0 else None)
            sel.to_csv(out_dir / f"learning_curves_{name}_cv.csv", index=False)
        ax1.set_title(f"{name}: cross-validation (selected setting)", fontsize=9)
        for seed, g in final.groupby("seed"):
            ax2.plot(g["epoch"], g["train_loss"], lw=1, label=f"seed {seed}")
        final.to_csv(out_dir / f"learning_curves_{name}_final.csv", index=False)
        ax2.set_title(f"{name}: final training on all development trials", fontsize=9)
        for ax in (ax1, ax2):
            ax.set_xlabel("epoch")
            ax.set_ylabel("weighted error (loss, cm-based)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        save_figure(fig, out_dir / f"learning_curves_{name}.png")


def plot_breakdown(breakdown: pd.DataFrame, by: str, models: Sequence[str], unit: str, out_dir: Path) -> None:
    if breakdown.empty:
        return
    breakdown.to_csv(out_dir / f"breakdown_by_{by}.csv", index=False)
    groups = sorted(breakdown[by].unique())
    fig, axes = new_figure(4.2 * len(groups), 4.0, ncols=len(groups), squeeze=False, sharey=True)
    for c, grp in enumerate(groups):
        ax = axes[0][c]
        for i, model in enumerate(models):
            g = breakdown[(breakdown[by] == grp) & (breakdown["model"] == model)].sort_values("horizon_ms")
            ax.plot(g["horizon_ms"], g["mean_error"], marker=MARKERS[i % len(MARKERS)], color=model_colour(model, i), label=model)
        n = int(breakdown.loc[breakdown[by] == grp, "n_trials"].max())
        ax.set_title(f"{by} = {grp}  (n = {n} trials)", fontsize=9)
        ax.set_xlabel("horizon (ms)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"mean error ({unit})")
    axes[0][0].legend(fontsize=7)
    save_figure(fig, out_dir / f"breakdown_by_{by}.png")


def plot_overfitting(diag: pd.DataFrame, unit: str, out_dir: Path) -> None:
    if diag.empty:
        return
    diag.to_csv(out_dir / "overfitting_diagnostic.csv", index=False)
    fig, ax = new_figure(7.0, 4.5)
    for i, model in enumerate(sorted(diag["model"].unique())):
        g = diag[diag["model"] == model].sort_values("horizon_ms")
        ax.plot(g["horizon_ms"], g["dev_mean"], ls="--", marker="o", color=model_colour(model, i), label=f"{model}: training (dev) trials")
        ax.plot(g["horizon_ms"], g["test_mean"], ls="-", marker="s", color=model_colour(model, i), label=f"{model}: test trials")
    ax.set_xlabel("prediction horizon (ms)")
    ax.set_ylabel(f"mean error ({unit})")
    ax.set_title("Overfitting check: a large gap between training and test error = memorising", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save_figure(fig, out_dir / "overfitting_diagnostic.png")
