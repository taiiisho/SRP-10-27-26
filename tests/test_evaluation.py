"""End-to-end evaluation tests: no future leakage, identical frames, full synthetic pipeline."""

import numpy as np
import pandas as pd
import pytest

from handpredict.evaluation.predictions import predict_trial
from handpredict.kinematics import make_filter
from handpredict.ml.windows import WindowSpec
from handpredict.physics import make_physics_predictors
from handpredict.synthetic import make_trial_data, sample_reach_spec
from handpredict.trial import TrialData, process_trial

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def ml_predictor(cfg, tmp_path_factory):
    """An untrained (random) GRU exported to ONNX: enough to test plumbing and causality."""
    from handpredict.ml.export import export_onnx
    from handpredict.ml.inference import OnnxNetwork, WindowPredictor
    from handpredict.ml.normalize import Normalizer
    from handpredict.ml.train import seeded_network
    from handpredict.ml.windows import build_windows

    spec = WindowSpec.from_config(cfg, 60.0)
    rng = np.random.default_rng(0)
    data = make_trial_data(sample_reach_spec(rng, cfg["synthetic"], "T1", "medium", 0.5), cfg, rng, "x", "medium", "T1", noise_px=1)
    norm = Normalizer.fit(build_windows(data, spec), spec)
    net = seeded_network("recurrent", spec, {"cell": "gru", "hidden_size": 8, "head_hidden": 8}, 0)
    path = export_onnx(net, spec, tmp_path_factory.mktemp("onnx") / "gru.onnx")
    return spec, {"gru": WindowPredictor("gru", OnnxNetwork(path), norm, "cpu")}


def _trial(cfg, seed=4, speed="medium"):
    rng = np.random.default_rng(seed)
    spec = sample_reach_spec(rng, cfg["synthetic"], "T2", speed, 0.5)
    return make_trial_data(spec, cfg, rng, f"trial{seed}", speed, "T2", noise_px=1.5, miss_probability=0.02)


def test_no_prediction_uses_future_samples(cfg, ml_predictor):
    """Change EVERYTHING after frame j: no prediction made at frames <= j may change (any model)."""
    spec, ml = ml_predictor
    physics = make_physics_predictors(cfg)
    data = _trial(cfg)
    filt = make_filter(cfg)
    base = predict_trial(process_trial(data, filt, cfg), physics, ml, cfg["horizons_ms"], spec, "test", 0.07)
    rng = np.random.default_rng(9)
    for j in (60, 85, 110):
        xy = data.xy.copy()
        xy[j + 1:] += rng.normal(0, 8.0, xy[j + 1:].shape)  # a completely different future
        changed = TrialData(data.context, data.t, xy, data.measured, data.interpolated)
        other = predict_trial(process_trial(changed, filt, cfg), physics, ml, cfg["horizons_ms"], spec, "test", 0.07)
        cols = [c for c in base.columns if c.endswith(("_pred_x", "_pred_y"))]
        a = base[base["frame_index"] <= j].set_index(["frame_index", "horizon_ms"])[cols]
        b = other[other["frame_index"] <= j].set_index(["frame_index", "horizon_ms"])[cols]
        pd.testing.assert_frame_equal(a, b)  # bit-for-bit identical
        later = base[base["frame_index"] > j + 5]
        assert not later.empty


def test_every_model_is_scored_on_the_same_frames(cfg, ml_predictor):
    spec, ml = ml_predictor
    table = predict_trial(process_trial(_trial(cfg, 6), make_filter(cfg), cfg), make_physics_predictors(cfg), ml,
                          cfg["horizons_ms"], spec, "test", 0.07)
    error_cols = [c for c in table.columns if c.endswith("_error")]
    for col in error_cols:
        np.testing.assert_array_equal(table[col].notna().to_numpy(), table["gt_valid"].to_numpy(), err_msg=col)
    assert table["win_movement_strict"].sum() <= table["win_movement_any_target"].sum() <= table["win_whole_trial"].sum()
    strict = table[table["win_movement_strict"]]
    assert (strict["in_movement"]).all()


@pytest.mark.slow
def test_full_synthetic_pipeline(cfg, tmp_path):
    """record -> build-dataset -> train -> evaluate on synthetic data, in well under a few minutes."""
    from handpredict.config import apply_overrides
    from handpredict.selftest import run_selftest

    tiny = apply_overrides(cfg, [
        "ml.training.max_epochs=4", "ml.training.patience=2", "ml.training.seeds=[0,1]", "ml.training.cv_folds=2",
        "ml.models.gru.hidden_size=[8]", "ml.models.gru.head_hidden=8", "ml.models.mlp.hidden_sizes=[[16]]",
        "ml.models.mlp.dropout=[0.0]", "ml.models.gp.max_train_points=80", "ml.models.gp.n_restarts_optimizer=0",
        "evaluation.bootstrap_samples=200",
    ])
    result = run_selftest(tiny, tmp_path, n_trials=18, quick=False)
    summary = result["summary"].read_text()
    assert "SYNTHETIC DATA" in summary and "## Verdict" in summary
    pred = pd.read_csv(result["results"] / "predictions.csv")
    required = ["trial_id", "participant_id", "speed_condition", "split", "timestamp", "x", "y", "vx", "vy", "ax", "ay",
                "horizon_ms", "physics_pred_x", "physics_pred_y", "ml_pred_x", "ml_pred_y", "actual_x", "actual_y",
                "physics_error", "ml_error", "physics_error_px", "ml_error_px", "zero_motion_error"]
    assert not set(required) - set(pred.columns)
    assert set(pred["split"]) == {"test", "dev"}
    assert (result["results"] / "plots" / "difference_vs_horizon_movement_strict.png").exists()
    log = pd.read_csv(result["results"] / "test_set_access_log.csv")
    assert len(log) == 1
