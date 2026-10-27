"""Tests of the ML window builder and the normaliser."""

import numpy as np
import pytest

from handpredict.config import apply_overrides
from handpredict.ml.normalize import Normalizer
from handpredict.ml.windows import WindowSpec, build_windows, time_scaled_copy, window_at
from handpredict.trial import TrialContext, TrialData


def _linear_trial(n=120, fps=60.0, v=(30.0, -12.0), target=(50.0, 5.0)):
    t = np.arange(n) / fps
    xy = np.array([10.0, 20.0]) + np.outer(t, v)
    ctx = TrialContext("tr", "s", "p", "fast", "T1", np.array([10.0, 20.0]), np.array(target), "cm", fps)
    return TrialData(ctx, t, xy, np.ones(n, bool))


def test_window_shapes_follow_the_config(cfg):
    spec = WindowSpec.from_config(cfg, fps=60.0)
    assert spec.n_samples == 12  # 200 ms of history at 60 fps
    ws = build_windows(_linear_trial(), spec)
    n = 120 - spec.n_samples + 1
    assert ws.seq.shape == (n, 12, 3)
    assert ws.static.shape == (n, 4)  # absolute position (2) + target relative to the hand (2)
    assert ws.target.shape == (n, len(cfg["horizons_ms"]), 2) and ws.mask.shape == (n, len(cfg["horizons_ms"]))
    np.testing.assert_array_equal(ws.seq[:, -1, :2], 0.0)  # positions are relative to "now"
    np.testing.assert_allclose(ws.seq[:, 1:, 2], 1 / 60.0)  # per-sample time steps


def test_window_answers_are_exact_on_linear_motion(cfg):
    spec = WindowSpec.from_config(cfg, fps=60.0)
    trial = _linear_trial()
    ws = build_windows(trial, spec)
    h = np.asarray(spec.horizons_s)
    expected = np.einsum("h,k->hk", h, np.array([30.0, -12.0]))
    for i in np.where(ws.mask.all(axis=1))[0]:
        np.testing.assert_allclose(ws.target[i], expected, atol=1e-9)
    # near the end of the recording, the long horizons have no ground truth
    assert not ws.mask[-1, -1] and ws.mask[0].all()
    j = int(ws.frame_index[0])
    np.testing.assert_allclose(ws.static[0, :2], trial.xy[j])
    np.testing.assert_allclose(ws.static[0, 2:], trial.context.target_xy - trial.xy[j])


def test_window_uses_only_the_past(cfg, rng):
    spec = WindowSpec.from_config(cfg, fps=60.0)
    trial = _linear_trial()
    xy2 = trial.xy.copy()
    xy2[60:] += rng.normal(0, 10, xy2[60:].shape)  # change the future
    for j in range(20, 60):
        a = window_at(trial.t, trial.xy, trial.measured, j, spec, trial.context.target_xy)
        b = window_at(trial.t, xy2, trial.measured, j, spec, trial.context.target_xy)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_no_window_across_a_long_gap_or_without_detection(cfg):
    spec = WindowSpec.from_config(cfg, fps=60.0)
    trial = _linear_trial()
    trial.xy[50:55] = np.nan
    trial.measured[50:55] = False
    ws = build_windows(trial, spec)
    assert not set(range(50, 55 + spec.n_samples - 1)) & set(ws.frame_index.tolist())
    assert set(ws.trial_id) == {"tr"}  # windows never mix trials


def test_ablation_flags_change_the_inputs(cfg):
    no_abs = WindowSpec.from_config(apply_overrides(cfg, ["ml.features.absolute_position=false"]), 60.0)
    no_target = WindowSpec.from_config(apply_overrides(cfg, ["ml.features.target=none"]), 60.0)
    assert no_abs.n_static == 2 and no_target.n_static == 2
    ws = build_windows(_linear_trial(), no_abs)
    np.testing.assert_allclose(ws.static[0], np.array([50.0, 5.0]) - _linear_trial().xy[ws.frame_index[0]])


def test_time_scaled_copy_plays_the_movement_faster(rng):
    trial = _linear_trial(n=121)
    fast = time_scaled_copy(trial, 2.0, rng)
    assert fast.t[-1] == pytest.approx(trial.t[-1] / 2, abs=1 / 60)
    np.testing.assert_allclose(fast.xy[10], trial.xy[0] + 2.0 * np.array([30.0, -12.0]) * fast.t[10], atol=1e-9)


def test_normaliser_round_trip_and_weights(cfg):
    spec = WindowSpec.from_config(cfg, fps=60.0)
    ws = build_windows(_linear_trial(), spec)
    norm = Normalizer.fit(ws, spec)
    seq_n, static_n = norm.inputs(ws.seq, ws.static)
    assert seq_n.dtype == np.float32 and np.isfinite(seq_n).all()
    np.testing.assert_allclose(norm.outputs_to_cm(norm.outputs_from_cm(ws.target)), ws.target, atol=1e-9)
    assert norm.horizon_weights.mean() == pytest.approx(1.0)
    assert norm.horizon_weights[0] > norm.horizon_weights[-1]  # short horizons get more weight
    again = Normalizer.from_dict(norm.to_dict())
    np.testing.assert_allclose(again.out_scale, norm.out_scale)
