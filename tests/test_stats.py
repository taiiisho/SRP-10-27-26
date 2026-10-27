"""Tests of the metrics and statistics on hand-computed cases."""

import numpy as np
import pandas as pd
import pytest

from handpredict.evaluation.metrics import bootstrap_ci, model_level, seed_variability, trial_level
from handpredict.evaluation.stats import (
    classify_difference_curve,
    crossover_analysis,
    holm,
    min_possible_p,
    paired_comparison,
    rank_biserial,
    trend_test,
    wilcoxon_test,
)
from handpredict.evaluation.summary import hypothesis_verdict


def test_holm_matches_hand_computation():
    p = [0.01, 0.04, 0.03, 0.005, np.nan]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone: 0.02, 0.03, 0.06, 0.06
    np.testing.assert_allclose(holm(p)[:4], [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm(p)[4])


def test_rank_biserial_and_wilcoxon():
    assert rank_biserial(np.array([-1.0, -2.0, -3.0])) == -1.0  # ML better on every trial
    assert rank_biserial(np.array([1.0, -2.0])) == pytest.approx((1 - 2) / 3)
    stat, p = wilcoxon_test(np.full(8, -0.5) + np.arange(8) * 0.01)
    assert p == pytest.approx(2 / 2**8)  # exact test, all signs equal
    assert np.isnan(wilcoxon_test(np.zeros(5))[1])
    assert min_possible_p(9) == pytest.approx(2 / 512)


def test_bootstrap_ci_is_reproducible_and_covers_the_mean():
    values = np.random.default_rng(0).normal(5.0, 1.0, 200)
    a = bootstrap_ci(values, lambda m: m.mean(axis=1), 2000, 0.95, np.random.default_rng(1))
    b = bootstrap_ci(values, lambda m: m.mean(axis=1), 2000, 0.95, np.random.default_rng(1))
    assert a == b and a[0] < values.mean() < a[1]
    assert a[1] - a[0] == pytest.approx(2 * 1.96 / np.sqrt(200), rel=0.2)


@pytest.mark.parametrize(
    "diff,expected",
    [
        ([0.4, 0.2, -0.2, -0.6], "crossover"),
        ([-0.1, -0.5, -1.0, -2.0], "ml_better_at_all_horizons"),
        ([0.1, 0.5, 1.0, 2.0], "physics_better_at_all_horizons"),
        ([-0.4, -0.2, 0.2, 0.6], "reverse_crossover"),
        ([0.4, -0.2, 0.2, -0.6], "multiple_crossings"),
    ],
)
def test_difference_curve_classification(diff, expected):
    assert classify_difference_curve([50, 100, 200, 500], diff)["classification"] == expected


def test_crossover_is_interpolated_linearly():
    result = classify_difference_curve([100, 200, 300], [0.3, 0.1, -0.1])
    assert result["first_crossing_ms"] == pytest.approx(250.0)


def _trial_table(diffs: dict[float, list[float]], physics=2.0):
    rows = []
    for h, ds in diffs.items():
        for i, d in enumerate(ds):
            for model, value in (("phys", physics), ("ml", physics + d)):
                rows.append({"window": "w", "model": model, "horizon_ms": h, "trial_id": f"t{i}", "participant_id": "p",
                             "speed_condition": "fast", "target_id": "T1", "split": "test", "n_frames": 10,
                             "mean": value, "median": value, "rmse": value, "p95": value})
    return pd.DataFrame(rows)


def test_paired_comparison_finds_a_consistent_difference():
    rng = np.random.default_rng(3)
    diffs = {50.0: list(0.5 + 0.05 * rng.normal(size=12)), 500.0: list(-1.0 + 0.05 * rng.normal(size=12))}
    table = paired_comparison(_trial_table(diffs), "w", "test", "phys", "ml", 500, 0.95, 0.05, 5, seed=1)
    row50, row500 = table.set_index("horizon_ms").loc[50.0], table.set_index("horizon_ms").loc[500.0]
    assert row50["better_model"] == "physics" and row500["better_model"] == "ml"
    assert row500["mean_diff"] == pytest.approx(np.mean(diffs[500.0]))
    assert row500["frac_trials_ml_better"] == 1.0 and row500["rank_biserial"] == -1.0
    assert row50["p_holm"] == pytest.approx(min(1, 2 * 2 / 2**12))


def test_crossover_bootstrap_and_trend():
    rng = np.random.default_rng(5)
    horizons = [50.0, 100.0, 200.0, 500.0]
    base = np.array([0.5, 0.2, -0.3, -1.5])
    dmat = pd.DataFrame(base + 0.05 * rng.normal(size=(15, 4)), columns=horizons)
    result = crossover_analysis(dmat, 500, 0.95, seed=2)
    assert result["observed"]["classification"] == "crossover"
    assert result["bootstrap"]["pattern_fractions"]["crossover"] > 0.95
    lo, hi = result["bootstrap"]["crossover_ci_ms"]
    assert 100 < lo <= result["observed"]["first_crossing_ms"] <= hi < 200
    trend = trend_test(dmat, 500, 0.95, seed=2)
    assert trend["median_slope_cm_per_100ms"] < 0 and trend["p_value"] < 0.01


def test_trial_level_metrics_by_hand():
    pred = pd.DataFrame({
        "trial_id": ["a"] * 4, "participant_id": "p", "speed_condition": "slow", "target_id": "T1", "split": "test",
        "horizon_ms": [50.0] * 4, "win_w": [True, True, True, False],
        "m_error": [1.0, 2.0, 4.0, 100.0],
    })
    out = trial_level(pred, ["m"], ["w"]).iloc[0]
    assert out["n_frames"] == 3  # the frame outside the window is ignored
    assert out["mean"] == pytest.approx(7 / 3)
    assert out["median"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(np.sqrt((1 + 4 + 16) / 3))
    assert out["p95"] == pytest.approx(np.quantile([1.0, 2.0, 4.0], 0.95))


def test_model_level_averages_trials_equally_and_seed_variability():
    rows = []
    for model, values in (("gru@seed0", [1.0, 3.0]), ("gru@seed1", [2.0, 4.0])):
        for i, v in enumerate(values):
            rows.append({"window": "w", "model": model, "horizon_ms": 50.0, "trial_id": f"t{i}", "split": "test",
                         "n_frames": 10 if i == 0 else 1000, "mean": v, "median": v, "rmse": v, "p95": v})
    stats = model_level(pd.DataFrame(rows), "test", 200, 0.95, seed=0)
    assert stats.set_index("model").loc["gru@seed0", "mean"] == 2.0  # trial-weighted, not frame-weighted
    sv = seed_variability(stats).iloc[0]
    assert sv["seed_mean"] == pytest.approx(2.5) and sv["seed_sd"] == pytest.approx(np.std([2.0, 3.0], ddof=1))


def _paired(diffs_by_h: dict, p_holm: dict, n=12):
    rows = []
    for h, d in diffs_by_h.items():
        rows.append({"horizon_ms": h, "n_trials": n, "mean_diff": d, "p_holm": p_holm[h], "significant": p_holm[h] < 0.05,
                     "min_possible_p_holm": 7 * 2 / 2**n, "tested": True})
    return pd.DataFrame(rows)


def test_verdict_rules():
    hs = [50.0, 100.0, 200.0, 500.0]

    def verdict(diffs, ps):
        paired = _paired(dict(zip(hs, diffs)), dict(zip(hs, ps)))
        cross = {"observed": classify_difference_curve(hs, diffs)}
        return hypothesis_verdict(paired, cross, {}, 0.05, 5)[0]

    assert verdict([0.5, 0.1, -0.5, -2.0], [0.01, 0.5, 0.2, 0.001]) == "supported"
    assert verdict([0.5, 0.1, -0.5, -2.0], [0.01, 0.5, 0.2, 0.3]) == "suggestive"
    assert verdict([-0.5, -0.6, -1.0, -2.0], [0.01, 0.01, 0.01, 0.01]) == "partially_supported"
    assert verdict([0.5, 0.6, 1.0, 2.0], [0.01, 0.01, 0.01, 0.01]) == "contradicted"
    assert verdict([0.5, 0.6, 1.0, 2.0], [0.5, 0.5, 0.5, 0.5]) == "unresolved"
    # too few trials for any test
    small = _paired(dict(zip(hs, [-1.0] * 4)), dict(zip(hs, [np.nan] * 4)), n=3)
    assert hypothesis_verdict(small, {"observed": {}}, {}, 0.05, 5)[0] == "unresolved"
