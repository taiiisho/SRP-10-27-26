"""The live (frame-by-frame) path must give exactly the same predictions as the analysis."""

import numpy as np

from handpredict.kinematics import make_filter
from handpredict.online import OnlinePredictor
from handpredict.physics import make_physics_predictors
from handpredict.synthetic import make_trial_data, sample_reach_spec
from handpredict.trial import make_history, process_trial


def test_online_predictions_equal_offline_predictions(cfg):
    rng = np.random.default_rng(21)
    spec = sample_reach_spec(rng, cfg["synthetic"], "T3", "medium", 0.5)
    data = make_trial_data(spec, cfg, rng, "live", "medium", "T3", noise_px=1.5, miss_probability=0.03)
    offline = process_trial(data, make_filter(cfg), cfg)
    physics = make_physics_predictors(cfg)
    online = OnlinePredictor(cfg, data.context, physics, trim=False)
    horizons = np.asarray(cfg["horizons_ms"], dtype=float) / 1000.0
    compared = 0
    for j in range(data.n):
        online.update(data.t[j], data.xy[j] if data.measured[j] else None)
        live = online.predict(horizons)
        if not live:
            assert not data.measured[j]
            continue
        hist = make_history(offline, j)
        # Same gap filling, sample by sample (vectorised vs scalar arithmetic may differ in the last bit).
        np.testing.assert_allclose(online.history().xy, hist.xy, rtol=0, atol=1e-9)
        for name, model in physics.items():
            np.testing.assert_allclose(live[name].positions, model.predict(hist, horizons).positions,
                                       rtol=0, atol=1e-9, err_msg=name)
        compared += 1
    assert compared > 150


def test_online_buffer_is_trimmed_after_a_movement(cfg):
    rng = np.random.default_rng(3)
    spec = sample_reach_spec(rng, cfg["synthetic"], "T1", "fast", 0.5)
    data = make_trial_data(spec, cfg, rng, "live", "fast", "T1", total_s=4.0)
    online = OnlinePredictor(cfg, data.context, {}, trim=True, rest_reset_s=0.6)
    for j in range(data.n):
        online.update(data.t[j], data.xy[j])
    assert online.t[0] > spec.onset_s + spec.duration_s  # the finished movement was dropped
    speed = np.linalg.norm(np.asarray(online.fv), axis=1)
    seg = cfg["kinematics"]["segmentation"]
    from handpredict.kinematics import find_onset

    assert find_onset(speed, seg["onset_speed_cm_s"], seg["onset_min_frames"]) is None  # only rest is left


def test_benchmark_and_headless_demo_run(cfg, tmp_path):
    from handpredict.benchmark import benchmark_pipeline, benchmark_predictors, write_report
    from handpredict.live_demo import run_demo

    table, _ = benchmark_predictors(cfg, None, iterations=20)
    assert set(cfg["physics"]["models"]) <= set(table["model"]) and (table["p95_ms"] > 0).all()
    pipe = benchmark_pipeline(cfg, "synthetic", "synthetic", duration_s=1.0, models_dir=None, calibration_path=None, video=None)
    assert pipe["frames"] > 20 and pipe["stages"]["end_to_end"]["n"] == pipe["frames"]
    report = write_report(tmp_path, table, pipe, {}, cfg).read_text()
    assert "Frame arrival -> both predictions ready" in report
    result = run_demo(cfg, "synthetic", "synthetic", display="headless", duration_s=2.0)
    assert result["frames"] > 30 and np.isfinite(result["rolling_errors_cm"][cfg["physics"]["headline"]])
