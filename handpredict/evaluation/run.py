"""``handpredict evaluate``: the single evaluation pipeline.

1. Load the processed dataset and the trained models (checking they belong together).
2. Predict with every model on the same frames of the TEST trials (and, as an
   overfitting diagnostic only, of the development trials).
3. Aggregate to trial level, compute statistics, plots and ``summary.md``.

Every run on the test set is appended to ``test_set_access_log.csv`` and the
count is printed in the summary, because every extra look at the test set
is a chance to (unconsciously) tune the study to it.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..config import config_hash
from ..dataset import MixedDataError, check_kinematics_match, load_dataset
from ..ml.inference import load_ml_predictors
from ..ml.windows import WindowSpec
from ..physics import make_physics_predictors
from ..utils import git_commit, hardware_description, now_iso, read_json, sha256_file, software_versions, write_json
from . import plots
from .metrics import fallback_rates, model_level, seed_variability, trial_level
from .predictions import add_headline_columns, build_prediction_table
from .stats import (
    crossover_analysis,
    paired_comparison,
    paired_matrix,
    summarise_breakdown,
    trend_test,
)
from .summary import (
    conditions_text,
    hypothesis_verdict,
    markdown_table,
    sequence_text,
    write_summary,
    _ci,
    _fmt,
    _p,
)

log = logging.getLogger(__name__)


def _ml_headline_name(cfg: Mapping, predictors: Mapping) -> str:
    name = cfg["ml"]["headline"]
    if name in predictors:
        return name
    seeded = [k for k in predictors if k.startswith(f"{name}@seed")]
    if not seeded:
        raise ValueError(f"headline ML model {name!r} was not found among the trained models {sorted(predictors)}")
    log.warning("No ensemble for %s; using %s as the ML headline", name, seeded[0])
    return seeded[0]


def run_evaluation(
    cfg: Mapping,
    dataset_dir: str | Path,
    models_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    split: str = "test",
    include_dev: bool = True,
    allow_mixed_synthetic: bool | None = None,
) -> Path:
    ev = cfg["evaluation"]
    dataset = load_dataset(dataset_dir)
    check_kinematics_match(dataset, cfg)
    allow_mixed = ev["allow_mixed_synthetic"] if allow_mixed_synthetic is None else allow_mixed_synthetic
    if dataset.data_origin == "mixed" and not allow_mixed:
        raise MixedDataError("this dataset mixes synthetic and real data; refusing to evaluate it")
    models_dir = Path(models_dir or Path(cfg["paths"]["models_dir"]) / dataset.name)
    training = read_json(models_dir / "training_summary.json")
    if training["dataset"] != dataset.name:
        raise ValueError(f"models in {models_dir} were trained on dataset {training['dataset']!r}, not {dataset.name!r}")
    caveats: list[str] = []
    if training["dataset_manifest_sha256"] != sha256_file(dataset.path / "dataset_manifest.json"):
        msg = "The dataset was rebuilt after the models were trained; retrain before trusting these results."
        log.warning(msg)
        caveats.append(msg)
    spec = WindowSpec.from_dict(training["window_spec"])
    horizons_ms = [float(h) for h in cfg["horizons_ms"]]
    if not np.allclose(np.asarray(spec.horizons_s) * 1000.0, horizons_ms):
        raise ValueError("the config's horizons differ from the ones the ML models were trained for; retrain")

    physics = make_physics_predictors(cfg)
    ml = load_ml_predictors(models_dir)
    physics_name = cfg["physics"]["headline"]
    ml_name = _ml_headline_name(cfg, ml)
    eval_ids = dataset.trial_ids(split)
    if not eval_ids:
        raise ValueError(f"the dataset has no {split} trials")
    trial_list = [(dataset.processed_trial(t), split) for t in eval_ids]
    if include_dev and split == "test":
        trial_list += [(dataset.processed_trial(t), "dev") for t in dataset.trial_ids("dev")]
    log.info("Evaluating %d models on %d %s trials (+%d development trials as a diagnostic)",
             len(physics) + len(ml), len(eval_ids), split, len(trial_list) - len(eval_ids))
    pred = build_prediction_table(trial_list, physics, ml, horizons_ms, spec, cfg["kinematics"]["ground_truth"]["max_gap_ms"] / 1000.0)
    pred = add_headline_columns(pred, physics_name, ml_name)

    out = Path(out_dir or Path(cfg["paths"]["results_dir"]) / (dataset.name if split == "test" else f"{dataset.name}_{split}"))
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv", index=False, float_format="%.6g")

    windows = list(ev["windows"])
    primary = ev["primary_window"]
    models = list(physics) + list(ml)
    seed = int(cfg["study"]["seed"])
    n_boot, level, alpha = int(ev["bootstrap_samples"]), float(ev["ci_level"]), float(ev["alpha"])
    min_trials = int(ev["min_trials_for_tests"])
    trials_df = trial_level(pred, models, windows)
    trials_df.to_csv(out / "trial_metrics.csv", index=False, float_format="%.6g")
    stats = model_level(trials_df, split, n_boot, level, seed)
    stats.to_csv(out / "model_metrics.csv", index=False, float_format="%.6g")

    paired = {w: paired_comparison(trials_df, w, split, physics_name, ml_name, n_boot, level, alpha, min_trials, seed) for w in windows}
    pd.concat(paired.values(), ignore_index=True).to_csv(out / "paired_tests.csv", index=False, float_format="%.6g")
    dmats = {w: paired_matrix(trials_df, w, split, physics_name, ml_name) for w in windows}
    crossover = {w: crossover_analysis(dmats[w], n_boot, level, seed) for w in windows}
    trend = trend_test(dmats[primary], n_boot, level, seed) if not dmats[primary].empty else {"n_trials": 0}
    write_json(out / "crossover.json", {"crossover": crossover, "trend_primary": trend}, overwrite=True)

    compare_models = [physics_name, ml_name, "zero_motion", "constant_velocity"]
    compare_models = [m for m in dict.fromkeys(compare_models) if m in models]
    breakdown_speed = summarise_breakdown(trials_df, primary, split, "speed_condition", compare_models)
    breakdown_target = summarise_breakdown(trials_df, primary, split, "target_id", compare_models)
    seeds_var = seed_variability(stats)
    seeds_var.to_csv(out / "seed_variability.csv", index=False, float_format="%.6g")
    fb = fallback_rates(pred[pred["split"] == split], list(physics), primary)
    fb.to_csv(out / "fallback_rates.csv", index=False)
    overfit = pd.DataFrame()
    if include_dev and split == "test" and (pred["split"] == "dev").any():
        dev_stats = model_level(trials_df, "dev", 200, level, seed, models=[ml_name, physics_name])
        test_stats = stats[stats["model"].isin([ml_name, physics_name])]
        overfit = dev_stats[["window", "model", "horizon_ms", "mean"]].rename(columns={"mean": "dev_mean"}).merge(
            test_stats[["window", "model", "horizon_ms", "mean"]].rename(columns={"mean": "test_mean"}),
            on=["window", "model", "horizon_ms"])
        overfit = overfit[overfit["window"] == primary]

    # -- plots ------------------------------------------------------------------
    unit = str(dataset.trials["units"].iloc[0])
    headline = [physics_name, ml_name]
    curve_models = [m for m in models if "@seed" not in m]
    key_models = [m for m in dict.fromkeys([physics_name, ml_name, "constant_velocity", "zero_motion"]) if m in models]
    for w in windows:
        plots.plot_error_vs_horizon(stats, w, key_models, headline, unit, plot_dir, split)
        plots.plot_error_vs_horizon(stats, w, curve_models, headline, unit, plot_dir, split,
                                    name=f"error_vs_horizon_all_models_{w}", log_y=True)
        plots.plot_difference_vs_horizon(paired[w], crossover[w], w, unit, plot_dir)
        plots.plot_paired_differences(dmats[w], w, unit, plot_dir)
    example_ids = _example_trials(pred, split)
    plots.plot_example_trajectories(pred, example_ids, ev["example_horizons_ms"], physics_name, ml_name, unit, plot_dir)
    speed_h = [h for h in (horizons_ms[0], float(ev["speed_plot_horizon_ms"]), horizons_ms[-1])]
    plots.plot_error_vs_speed(pred, primary, list(dict.fromkeys(speed_h)), compare_models, unit, plot_dir)
    plots.plot_learning_curves(models_dir, training, plot_dir)
    plots.plot_breakdown(breakdown_speed, "speed_condition", compare_models, unit, plot_dir)
    plots.plot_breakdown(breakdown_target, "target_id", compare_models, unit, plot_dir)
    plots.plot_overfitting(overfit, unit, plot_dir)

    # -- verdict and summary --------------------------------------------------------
    access_count = _log_test_access(out, cfg, models_dir, split)
    verdict_code, verdict_text = hypothesis_verdict(paired[primary], crossover[primary], trend, alpha, min_trials)
    ctx = _summary_context(
        cfg, dataset, training, split, primary, windows, stats, paired, crossover, trend, breakdown_speed, seeds_var, fb,
        overfit, pred, physics, ml, physics_name, ml_name, unit, verdict_code, verdict_text, access_count, caveats,
    )
    write_summary(out / "summary.md", ctx)
    write_json(out / "results_manifest.json", {
        "created": now_iso(), "split": split, "dataset": dataset.name, "dataset_path": str(dataset.path),
        "dataset_manifest_sha256": sha256_file(dataset.path / "dataset_manifest.json"),
        "models_dir": str(models_dir), "training_summary_sha256": sha256_file(models_dir / "training_summary.json"),
        "physics_headline": physics_name, "ml_headline": ml_name, "verdict": verdict_code,
        "ml_predictor_hardware": "CPU via ONNX Runtime (scikit-learn for the GP)",
        "evaluation_machine": hardware_description(), "config_hash": config_hash(cfg), "config": cfg,
        "software": software_versions(), "git_commit": git_commit(), "test_set_access_count": access_count,
    }, overwrite=True)
    log.info("Verdict: %s", verdict_text)
    log.info("Results written to %s (see summary.md)", out)
    return out


def _example_trials(pred: pd.DataFrame, split: str, n: int = 2) -> list[str]:
    """A slow and a fast test trial (or the first ones available) for the trajectory plot."""
    sub = pred[pred["split"] == split]
    chosen = []
    for speed in ("slow", "fast", "medium"):
        ids = sub.loc[sub["speed_condition"] == speed, "trial_id"].unique()
        if len(ids):
            chosen.append(ids[0])
        if len(chosen) == n:
            break
    if not chosen:
        chosen = list(sub["trial_id"].unique()[:n])
    return chosen


def _log_test_access(out: Path, cfg: Mapping, models_dir: Path, split: str) -> int:
    log_path = out / "test_set_access_log.csv"
    new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["time", "split", "config_hash", "training_summary_sha256", "git_commit"])
        writer.writerow([now_iso(), split, config_hash(cfg), sha256_file(models_dir / "training_summary.json")[:16], git_commit()])
    rows = pd.read_csv(log_path)
    return int((rows["split"] == "test").sum())


def _summary_context(cfg, dataset, training, split, primary, windows, stats, paired, crossover, trend, breakdown,
                     seeds_var, fb, overfit, pred, physics, ml, physics_name, ml_name, unit, verdict_code, verdict_text,
                     access_count, caveats) -> dict:
    horizons = [float(h) for h in cfg["horizons_ms"]]
    trials = dataset.trials[(dataset.trials["split"] == split) & dataset.trials["included"]]
    prim = pred[(pred["split"] == split) & pred[f"win_{primary}"]]
    frames = prim.groupby("horizon_ms")["frame_index"].count().reindex(horizons, fill_value=0).tolist()
    tph = prim.groupby("horizon_ms")["trial_id"].nunique().reindex(horizons, fill_value=0).tolist()
    st = stats[stats["window"] == primary].set_index(["model", "horizon_ms"])

    def stat(model, h, col):
        try:
            return float(st.loc[(model, h), col])
        except KeyError:
            return float("nan")

    main_rows = []
    pp = paired[primary].set_index("horizon_ms") if not paired[primary].empty else pd.DataFrame()
    for h in horizons:
        r = pp.loc[h] if h in pp.index else None
        main_rows.append({
            "horizon_ms": h, "n_trials": int(r["n_trials"]) if r is not None else 0,
            "physics_mean": stat(physics_name, h, "mean"), "physics_lo": stat(physics_name, h, "mean_ci_low"),
            "physics_hi": stat(physics_name, h, "mean_ci_high"),
            "ml_mean": stat(ml_name, h, "mean"), "ml_lo": stat(ml_name, h, "mean_ci_low"), "ml_hi": stat(ml_name, h, "mean_ci_high"),
            "zero_mean": stat("zero_motion", h, "mean"),
            "mean_diff": r["mean_diff"] if r is not None else np.nan,
            "diff_lo": r["mean_diff_ci_low"] if r is not None else np.nan,
            "diff_hi": r["mean_diff_ci_high"] if r is not None else np.nan,
            "p_holm": r["p_holm"] if r is not None else np.nan,
            "rank_biserial": r["rank_biserial"] if r is not None else np.nan,
            "better": r["better_model"] if r is not None else "-",
        })

    obs = crossover[primary]["observed"]
    boot = crossover[primary].get("bootstrap") or {}
    if obs["classification"] == "crossover":
        ci = boot.get("crossover_ci_ms")
        cross_text = (f"Observed crossover at **{obs['first_crossing_ms']:.0f} ms** (where the mean ML - physics difference "
                      f"changes from positive to negative)" + (f"; bootstrap 95% CI {ci[0]:.0f}-{ci[1]:.0f} ms." if ci else "."))
    else:
        cross_text = f"No crossover in the tested range (50-{horizons[-1]:g} ms): pattern = `{obs['classification']}`."
    if boot:
        fractions = ", ".join(f"{k}: {100 * v:.0f}%" for k, v in boot["pattern_fractions"].items())
        cross_text += f" When the test trials are resampled ({boot['n_boot']} bootstrap samples) the patterns appear as: {fractions}."

    if trend.get("n_trials", 0):
        ci = trend["median_slope_ci"]
        trend_text = (
            f"Per-trial slope of (ML - physics) error against horizon: median **{trend['median_slope_cm_per_100ms']:.3f} {unit} per "
            f"100 ms** (95% CI {_fmt(ci[0], 3)} to {_fmt(ci[1], 3)}; {100 * trend['frac_trials_negative_slope']:.0f}% of "
            f"{trend['n_trials']} trials negative; Wilcoxon p = {_p(trend['p_value'])}). A negative slope means ML gains on "
            "physics as the horizon grows, the direction the hypothesis predicts."
        )
    else:
        trend_text = "Not enough trials with several horizons to estimate a trend."

    curve_models = [m for m in list(physics) + list(ml) if "@seed" not in m]
    table = markdown_table(stats[(stats["window"] == primary) & stats["model"].isin(curve_models)],
                           "model", "horizon_ms", "mean")

    sec_lines = []
    for w in windows:
        if w == primary or paired[w].empty:
            continue
        cells = "; ".join(
            f"{r.horizon_ms:g} ms: {_fmt(r.mean_diff)} {_ci(r.mean_diff_ci_low, r.mean_diff_ci_high)}, p(Holm) {_p(r.p_holm)}"
            for r in paired[w].itertuples())
        sec_lines.append(f"- `{w}` (ML - physics, {unit}): {cells}. Pattern: `{crossover[w]['observed']['classification']}`.")
    secondary_text = "\n".join(sec_lines) or "None."

    req = [h for h in cfg["required_horizons_ms"]]
    bd = breakdown[breakdown["horizon_ms"].isin(req) & breakdown["model"].isin([physics_name, ml_name])]
    breakdown_text = markdown_table(bd.assign(key=bd["speed_condition"] + " / " + bd["model"]), "key", "horizon_ms", "mean_error") \
        if not bd.empty else "No data."

    model_lines = [
        f"Physics headline: `{physics_name}` (Flash & Hogan minimum-jerk path to the calibrated target, onset time and "
        "duration fitted online, rolled forward with Newton-Euler steps; constant-velocity fallback when the fit is not ready).",
    ]
    card = training["models"].get(ml_name.split("@")[0], {})
    cv = card.get("cv", {})
    model_lines.append(
        f"ML headline: `{ml_name}` ({card.get('kind')} network, hyperparameters {card.get('params')}, chosen by "
        f"{cv.get('folds', '?')}-fold grouped cross-validation on development trials (validation loss "
        f"{_fmt(cv.get('selected_mean_val_loss'), 3)} +/- {_fmt(cv.get('selected_sd_val_loss'), 3)}); average of "
        f"{len(card.get('seeds', []))} copies trained with different random seeds)."
    )
    others = [m for m in curve_models if m not in (physics_name, ml_name)]
    model_lines.append(f"Other rows (ablations / baselines, not alternatives to the headlines): {', '.join(f'`{m}`' for m in others)}.")
    detector = sorted({(s["controlled_variables"].get("detector_backend"), s["controlled_variables"].get("detector_hardware"),
                        s["controlled_variables"].get("detector_fallback_from")) for s in dataset.manifest["sessions"]})
    model_lines.append("Hand-landmark detection ran on: " + "; ".join(
        f"{b} on {hw}" + (f" (fallback from {fb_})" if fb_ else "") for b, hw, fb_ in detector) + ".")
    model_lines.append("ML predictor in this evaluation ran on: the CPU via ONNX Runtime (exactly the exported model files). "
                       "See `handpredict benchmark` for edge latency, including any Hailo run.")

    diagnostics = []
    if not seeds_var.empty:
        sv = seeds_var[(seeds_var["window"] == primary) & seeds_var["horizon_ms"].isin(req)]
        diagnostics.append("Seed variability (single trained copies, mean +/- sd over seeds): " + "; ".join(
            f"{r.base_model} {r.horizon_ms:g} ms: {_fmt(r.seed_mean)} +/- {_fmt(r.seed_sd)}" for r in sv.itertuples()))
    if not overfit.empty:
        o = overfit[(overfit["model"] == ml_name) & overfit["horizon_ms"].isin(req)]
        diagnostics.append("Overfitting check, ML error on training (dev) vs test trials: " + "; ".join(
            f"{r.horizon_ms:g} ms: {_fmt(r.dev_mean)} vs {_fmt(r.test_mean)}" for r in o.itertuples())
            + " (a much lower training error would mean memorising).")
    if not fb.empty:
        f = fb[fb["model"] == physics_name]
        diagnostics.append("Physics headline fell back to constant velocity in " + ", ".join(
            f"{100 * r.fallback_fraction:.0f}% ({r.horizon_ms:g} ms)" for r in f.itertuples() if r.horizon_ms in req)
            + " of primary-window predictions (mostly right after movement onset).")
    jitter = dataset.manifest.get("resting_jitter_dev_median")
    if jitter is not None:
        diagnostics.append(f"Detector jitter of a still hand: {jitter:.3f} {unit} (median, development trials). The ground "
                           "truth is a measurement too, so errors of this size cannot be told apart from noise.")

    n_trials = len(trials)
    fps = float(trials["fps"].median()) if n_trials else float("nan")
    all_caveats = list(caveats)
    if dataset.data_origin != "real":
        all_caveats.insert(0, "SYNTHETIC DATA: nothing here is evidence about real hands.")
    if n_trials < 30:
        all_caveats.append(f"Only {n_trials} {split} trials: confidence intervals are wide and small differences cannot be detected.")
    if trials["participant_id"].nunique() <= 1:
        all_caveats.append("All trials come from one participant, so the results may not generalise to other people.")
    if np.isfinite(fps):
        all_caveats.append(f"At {fps:g} fps the 50 ms horizon is only {50 * fps / 1000:.1f} frames ahead; timing jitter and "
                           "detection noise matter most there.")
    all_caveats.append("In the primary window, long horizons are only evaluated on slower movements (t + h must still be "
                       "inside the movement), so the horizon effect is partly mixed with a speed effect; compare with the "
                       "`movement_any_target` window.")
    all_caveats += [f"Controlled-variable warning: {w}" for w in dataset.manifest.get("controlled_variable_warnings", [])]
    if access_count > 1:
        all_caveats.append(f"The test set has now been evaluated {access_count} times; only the first evaluation is a "
                           "fully clean test (see test_set_access_log.csv).")
    return {
        "unit": unit, "data_origin": dataset.data_origin, "split": split, "created": now_iso(), "dataset": dataset.name,
        "git_commit": git_commit(), "config_hash": config_hash(cfg), "test_access_count": access_count,
        "verdict_code": verdict_code, "verdict_text": verdict_text, "primary_window": primary,
        "sample": {
            "n_trials": n_trials, "n_participants": int(trials["participant_id"].nunique()),
            "n_dev": int((dataset.trials["split"] == "dev").sum()), "conditions": conditions_text(trials),
            "frames_per_horizon": sequence_text(frames, horizons), "trials_per_horizon": sequence_text(tph, horizons),
        },
        "model_lines": model_lines, "physics": physics_name, "ml": ml_name, "main_rows": main_rows,
        "required_horizons": [float(h) for h in req], "crossover_text": cross_text, "trend_text": trend_text,
        "all_models_table": table, "secondary_text": secondary_text, "breakdown_text": breakdown_text,
        "diagnostics": diagnostics, "caveats": all_caveats,
    }
