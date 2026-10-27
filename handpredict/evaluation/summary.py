"""The automatically written ``summary.md``: numbers, sample size and a cautious verdict.

The verdict is produced by fixed rules (below), applied to the PRIMARY
analysis window on the TEST trials, so it cannot be nudged after seeing the
results. It deliberately says "unresolved" whenever the data cannot tell.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

VERDICT_TITLES = {
    "supported": "The data SUPPORT the hypothesis",
    "partially_supported": "The data PARTLY support the hypothesis",
    "suggestive": "The data point in the hypothesised direction but DO NOT RESOLVE it",
    "contradicted": "The data CONTRADICT the hypothesis",
    "unresolved": "The data DO NOT RESOLVE the hypothesis",
}


def _fmt(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{digits}f}"


def _ci(lo: Any, hi: Any, digits: int = 2) -> str:
    return f"[{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _p(p: Any) -> str:
    if p is None or not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def hypothesis_verdict(paired: pd.DataFrame, crossover: Mapping, trend: Mapping, alpha: float, min_trials: int) -> tuple[str, str]:
    """Rule-based verdict on 'as the horizon grows, ML eventually has lower error than physics'."""
    if paired.empty or paired["n_trials"].max() < min_trials:
        n = 0 if paired.empty else int(paired["n_trials"].max())
        return "unresolved", (
            f"Only {n} paired test trials are available (at least {min_trials} are needed for a test), so no "
            "statistical conclusion is possible. Record more trials."
        )
    reachable = bool((paired["min_possible_p_holm"] < alpha).any())
    obs = crossover.get("observed", {})
    cls = obs.get("classification")
    sig = paired[paired["significant"]]
    sig_ml = sig.loc[sig["mean_diff"] < 0, "horizon_ms"].tolist()
    sig_phys = sig.loc[sig["mean_diff"] > 0, "horizon_ms"].tolist()
    longest = paired["horizon_ms"].max()
    n_max = int(paired["n_trials"].max())
    if not reachable:
        return "unresolved", (
            f"With only {n_max} paired trials per horizon, even a perfectly consistent difference could not reach "
            f"significance after correcting for {int(paired['tested'].sum())} horizons (smallest possible Holm-adjusted p = "
            f"{paired['min_possible_p_holm'].min():.3f}). The observed pattern is '{cls}', but more trials are needed."
        )
    if cls == "crossover":
        cross = obs["first_crossing_ms"]
        after_ml = [h for h in sig_ml if h > cross]
        after_phys = [h for h in sig_phys if h > cross]
        if after_ml and not after_phys:
            return "supported", (
                f"Physics-based prediction was better at short horizons and ML at longer ones; the mean difference "
                f"changed sign at about {cross:.0f} ms. ML's advantage was statistically significant (Holm-corrected) at "
                f"{', '.join(f'{h:g}' for h in after_ml)} ms."
            )
        return "suggestive", (
            f"The mean difference changed sign at about {cross:.0f} ms (physics better before, ML better after), as "
            "hypothesised, but ML's advantage at the longer horizons is not statistically significant after "
            "correction, so chance cannot be ruled out."
        )
    if cls == "ml_better_at_all_horizons":
        if sig_ml:
            return "partially_supported", (
                "ML had lower error at EVERY tested horizon, including the shortest (50 ms), and the difference was "
                f"significant at {', '.join(f'{h:g}' for h in sig_ml)} ms. So ML does produce lower error at long horizons, "
                "but there is no crossover: physics was never better, which the word 'eventually' implied."
            )
        return "unresolved", "ML had lower mean error at every horizon, but no difference was statistically significant."
    if cls == "physics_better_at_all_horizons":
        if longest in sig_phys:
            return "contradicted", (
                f"Physics-based prediction had lower error at every tested horizon up to {longest:g} ms, significantly "
                f"so at {', '.join(f'{h:g}' for h in sig_phys)} ms. ML never produced lower error in the tested range."
            )
        return "unresolved", (
            "Physics had lower mean error at every horizon, but not significantly at the longest horizon; no crossover "
            "was observed in the tested range (50-500 ms)."
        )
    if cls == "reverse_crossover":
        if any(h >= obs["first_crossing_ms"] for h in sig_phys):
            return "contradicted", (
                "ML was better at short horizons but physics was better at long horizons: the opposite of the "
                "hypothesis, with physics significantly better at "
                f"{', '.join(f'{h:g}' for h in sig_phys)} ms."
            )
        return "unresolved", "ML was better at short horizons and physics at long ones, but not significantly."
    return "unresolved", f"The difference between the models changes sign more than once ({cls}); there is no consistent pattern."


def write_summary(path, ctx: Mapping[str, Any]) -> str:
    """Render summary.md from the evaluation context and write it to ``path``."""
    lines: list[str] = []
    add = lines.append
    unit = ctx["unit"]
    add("# Results summary: physics-based vs edge-AI prediction of hand movements")
    add("")
    if ctx["data_origin"] != "real":
        add("> **WARNING: THIS SUMMARY IS BASED ON SYNTHETIC DATA.** It only shows that the software works. "
            "It says NOTHING about real hand movements and must not be reported as a result.")
        add("")
    if ctx["split"] != "test":
        add("> **DEVELOPMENT DATA:** the ML models were trained on these trials. These numbers are a diagnostic, "
            "not a result.")
        add("")
    add(f"Generated {ctx['created']} | dataset `{ctx['dataset']}` | code `{ctx['git_commit']}` | config `{ctx['config_hash']}`"
        f" | evaluation of the test set number {ctx['test_access_count']}")
    add("")
    add(f"## Verdict: {VERDICT_TITLES[ctx['verdict_code']]}")
    add("")
    add(ctx["verdict_text"])
    add("")
    add(f"*Hypothesis: if the prediction horizon increases, the ML model will eventually produce lower trajectory error "
        f"than the physics-based model.* Decided by the rules in `handpredict/evaluation/summary.py`, applied to the "
        f"primary window (`{ctx['primary_window']}`) on the {ctx['split']} trials only.")
    add("")
    s = ctx["sample"]
    add("## Sample size")
    add("")
    add(f"- {ctx['split'].capitalize()} trials: **{s['n_trials']}** from **{s['n_participants']}** participant(s); "
        f"development (training) trials: {s['n_dev']}.")
    add(f"- Test trials per condition: {s['conditions']}.")
    add(f"- Evaluated frames per horizon (primary window): {s['frames_per_horizon']}.")
    add(f"- Trials with data per horizon (primary window): {s['trials_per_horizon']}. In the primary window a prediction "
        "only counts if t + horizon is still inside the movement, so fast movements drop out at long horizons; the "
        "`movement_any_target` window below does not have this speed bias.")
    add("")
    add("## Models (headlines fixed before looking at test results)")
    add("")
    for line in ctx["model_lines"]:
        add(f"- {line}")
    add("")
    add(f"## Main result ({ctx['primary_window']} window, {ctx['split']} trials, errors in {unit})")
    add("")
    add(f"| horizon (ms) | trials | physics: {ctx['physics']} [95% CI] | ML: {ctx['ml']} [95% CI] | zero-motion | "
        "ML - physics [95% CI] | p (Holm) | effect r | better |")
    add("|---:|---:|---|---|---:|---|---:|---:|---|")
    for r in ctx["main_rows"]:
        star = " **(required)**" if r["horizon_ms"] in ctx["required_horizons"] else ""
        add(f"| {r['horizon_ms']:g}{star} | {r['n_trials']} | {_fmt(r['physics_mean'])} {_ci(r['physics_lo'], r['physics_hi'])} | "
            f"{_fmt(r['ml_mean'])} {_ci(r['ml_lo'], r['ml_hi'])} | {_fmt(r['zero_mean'])} | "
            f"{_fmt(r['mean_diff'])} {_ci(r['diff_lo'], r['diff_hi'])} | {_p(r['p_holm'])} | {_fmt(r['rank_biserial'])} | {r['better']} |")
    add("")
    add("Effect r = matched-pairs rank-biserial correlation (-1 = ML better on every trial, +1 = physics better on every "
        "trial). p-values: Wilcoxon signed-rank on per-trial mean errors, Holm-corrected across horizons.")
    add("")
    add("## Crossover")
    add("")
    add(ctx["crossover_text"])
    add("")
    add("## Trend with horizon")
    add("")
    add(ctx["trend_text"])
    add("")
    add(f"## Mean error of every model ({ctx['primary_window']}, {ctx['split']} trials, {unit})")
    add("")
    add(ctx["all_models_table"])
    add("")
    add("## Secondary analysis windows")
    add("")
    add(ctx["secondary_text"])
    add("")
    add("## Breakdown by speed condition (descriptive only, no tests)")
    add("")
    add(ctx["breakdown_text"])
    add("")
    add("## Diagnostics")
    add("")
    for line in ctx["diagnostics"]:
        add(f"- {line}")
    add("")
    add("## Caveats")
    add("")
    for line in ctx["caveats"]:
        add(f"- {line}")
    add("")
    add("## Files")
    add("")
    add("`predictions.csv` (every prediction), `trial_metrics.csv` (per trial), `model_metrics.csv` (per model, with "
        "bootstrap CIs), `paired_tests.csv`, `crossover.json`, `plots/` (PNG figures + the CSV behind each).")
    add("")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def markdown_table(df: pd.DataFrame, index: str, columns: str, values: str, digits: int = 2) -> str:
    wide = df.pivot_table(index=index, columns=columns, values=values)
    header = f"| {index} | " + " | ".join(f"{c:g}" if isinstance(c, (int, float)) else str(c) for c in wide.columns) + " |"
    sep = "|---|" + "---:|" * len(wide.columns)
    rows = [f"| {i} | " + " | ".join(_fmt(v, digits) for v in row) + " |" for i, row in wide.iterrows()]
    return "\n".join([header, sep, *rows])


def conditions_text(trials: pd.DataFrame) -> str:
    counts = trials.groupby(["speed_condition", "target_id"]).size()
    return ", ".join(f"{s}/{t}: {n}" for (s, t), n in counts.items())


def sequence_text(values: Sequence, horizons: Sequence) -> str:
    return ", ".join(f"{h:g} ms: {v}" for h, v in zip(horizons, values))
