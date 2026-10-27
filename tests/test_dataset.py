"""Tests of build-dataset and the train pipeline's test-set isolation."""

import numpy as np
import pandas as pd
import pytest

from handpredict.config import apply_overrides
from handpredict.dataset import MixedDataError, build_dataset, compare_controlled_variables, load_dataset
from handpredict.kinematics import make_filter
from handpredict.record_session import run_recording_session
from handpredict.trial import process_trial

TINY_TRAINING = [
    "ml.training.max_epochs=3", "ml.training.patience=2", "ml.training.seeds=[0]", "ml.training.cv_folds=2",
    "ml.models.gru.hidden_size=[8]", "ml.models.gru.head_hidden=8", "ml.models.mlp.hidden_sizes=[[8]]",
    "ml.models.mlp.dropout=[0.0]", "ml.models.gp.max_train_points=60", "ml.models.gp.n_restarts_optimizer=0",
]


@pytest.fixture(scope="module")
def raw_dir(cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    for seed in (1, 2):
        run_recording_session(cfg, "SYN01", "synthetic", "synthetic", auto=True, n_trials=9, raw_dir=root, seed=seed,
                              session_id=f"synthetic_S{seed}_SYN01")
    return root


@pytest.fixture(scope="module")
def dataset_dir(cfg, raw_dir, tmp_path_factory):
    return build_dataset(cfg, raw_dir=raw_dir, out_root=tmp_path_factory.mktemp("processed"), plots=True)


def test_dataset_files_and_manifest(dataset_dir):
    ds = load_dataset(dataset_dir)
    assert ds.name.startswith("synthetic") and ds.data_origin == "synthetic"
    assert ds.manifest["n_trials_included"] == 18 and len(ds.manifest["sessions"]) == 2
    assert ds.manifest["n_test"] + ds.manifest["n_dev"] == 18
    assert set(ds.trials["split"]) <= {"dev", "test"}
    for col in ("t_s", "x", "y", "vx", "vy", "ax", "ay"):  # per-frame record required by the proposal
        assert col in ds.frames.columns
    assert len(list((dataset_dir / "segmentation_plots").glob("*.png"))) == 18


def test_reloaded_trial_matches_fresh_processing(cfg, dataset_dir):
    ds = load_dataset(dataset_dir)
    pt = ds.processed_trial(ds.trial_ids("dev")[0])
    fresh = process_trial(pt.data, make_filter(cfg), cfg)
    np.testing.assert_allclose(pt.kin.velocity, fresh.kin.velocity, rtol=1e-6, atol=1e-6, equal_nan=True)
    assert pt.segmentation.onset_index == fresh.segmentation.onset_index


def test_mixing_synthetic_and_real_is_refused(cfg, raw_dir, tmp_path):
    from handpredict.capture.synthetic_source import SyntheticSource
    from handpredict.landmarks.synthetic_detector import SyntheticDetector
    from handpredict.record_session import synthetic_calibration
    from handpredict.recording import SessionWriter

    # A session that claims to be real (only possible by bypassing the normal labelling).
    fake_real = SessionWriter(cfg, "P01", SyntheticSource(cfg), SyntheticDetector(cfg), synthetic_calibration(cfg),
                              raw_dir=raw_dir, session_id="S_fake_real_P01", synthetic=False)
    fake_real.finish()
    sessions = [p for p in raw_dir.iterdir()]
    with pytest.raises(MixedDataError):
        build_dataset(cfg, session_dirs=sessions, out_root=tmp_path)
    for p in fake_real.dir.iterdir():  # clean up so other tests see only synthetic sessions
        p.chmod(0o644)
        p.unlink()
    fake_real.dir.rmdir()


def test_controlled_variable_differences_are_reported():
    base = {"session_id": "A", "camera": {"width": 1280, "height": 720, "nominal_fps": 60, "source": "picamera2"},
            "detector": {"backend": "hailo", "hardware": "hailo-10h"}, "landmark": {"index": 9},
            "calibration": {"cm_per_px": 0.08}}
    other = {**base, "session_id": "B", "camera": {**base["camera"], "nominal_fps": 30},
             "detector": {"backend": "mediapipe", "hardware": "cpu", "fallback_from": "hailo"}}
    warnings = " ".join(compare_controlled_variables([base, other]))
    assert "nominal_fps" in warnings and "detector_backend" in warnings and "detector_fallback_from" in warnings


def test_train_pipeline_never_loads_test_trials(cfg, dataset_dir, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from handpredict import dataset as dataset_module
    from handpredict.ml.inference import load_ml_predictors
    from handpredict.ml.pipeline import train_all

    ds = load_dataset(dataset_dir)
    test_ids = set(ds.trial_ids("test"))
    original = dataset_module.ProcessedDataset.processed_trial

    def guarded(self, trial_id):
        assert trial_id not in test_ids, f"training touched test trial {trial_id}"
        return original(self, trial_id)

    monkeypatch.setattr(dataset_module.ProcessedDataset, "processed_trial", guarded)
    tiny = apply_overrides(cfg, TINY_TRAINING)
    out = train_all(tiny, dataset_dir, models_root=tmp_path)
    predictors = load_ml_predictors(out)
    assert {"gru", "gru@seed0", "mlp", "mlp@seed0", "gp"} <= set(predictors)
    summary = pd.read_json(out / "training_summary.json", typ="series")
    assert summary["n_test_trials_untouched"] == len(test_ids)
    assert not set(summary["dev_trials"]) & test_ids

    # Hailo export step 1: one flat input of fixed size; must reproduce the ONNX ensemble exactly.
    import onnxruntime as ort

    from handpredict.ml.hailo import export_onnx_for_hailo

    export_dir = export_onnx_for_hailo(tiny, out, "mlp")
    calib = np.load(export_dir / "calibration.npy")
    flat = ort.InferenceSession(str(export_dir / "mlp_flat.onnx"), providers=["CPUExecutionProvider"])
    mlp = predictors["mlp"]
    w = mlp.spec.n_samples * 3
    expected = mlp.run(calib[:20, :w].reshape(20, mlp.spec.n_samples, 3), calib[:20, w:])
    got = np.concatenate([flat.run(None, {"flat": calib[i : i + 1]})[0] for i in range(20)]).reshape(expected.shape)
    np.testing.assert_allclose(got, expected, atol=1e-5)
