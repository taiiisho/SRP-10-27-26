"""Tests for causal filtering, gap filling, segmentation and ground truth."""

import numpy as np
import pytest

from handpredict.kinematics import (
    CausalPolyfitFilter,
    KalmanCAFilter,
    fill_short_gaps,
    find_onset,
    ground_truth_at,
    make_filter,
    segment_movement,
)

FILTERS = [
    KalmanCAFilter(jerk_psd=2e6, measurement_std=0.15),
    CausalPolyfitFilter(window=9, order=2),
]


def _noisy_reach(rng, n=180, fps=60.0):
    t = np.cumsum(np.full(n, 1.0 / fps) + rng.normal(0, 0.0005, n))
    t -= t[0]
    s = np.clip((t - 0.8) / 0.9, 0, 1)
    s = s**3 * (10 - 15 * s + 6 * s**2)
    xy = np.stack([10 + 35 * s, 20 - 15 * s], axis=1) + rng.normal(0, 0.15, (n, 2))
    measured = rng.random(n) > 0.05
    return t, xy, measured


@pytest.mark.parametrize("filt", FILTERS, ids=lambda f: f.name)
def test_filter_is_causal(filt, rng):
    """Changing a future sample must not change any earlier output (no data leak)."""
    t, xy, measured = _noisy_reach(rng)
    base = filt.run(t, xy, measured)
    for k in (20, 75, 140):
        perturbed = xy.copy()
        perturbed[k:] += rng.normal(0, 5.0, perturbed[k:].shape)  # change the whole future
        out = filt.run(t, perturbed, measured)
        for name in ("position", "velocity", "acceleration"):
            np.testing.assert_array_equal(getattr(out, name)[:k], getattr(base, name)[:k])


@pytest.mark.parametrize("filt", FILTERS, ids=lambda f: f.name)
def test_filter_ignores_unmeasured_samples(filt, rng):
    """Values at frames flagged as not measured (e.g. interpolated) must not influence the output."""
    t, xy, measured = _noisy_reach(rng)
    measured[50:53] = False
    a = filt.run(t, xy, measured)
    xy2 = xy.copy()
    xy2[50:53] = 999.0
    b = filt.run(t, xy2, measured)
    np.testing.assert_array_equal(a.position, b.position)


@pytest.mark.parametrize("filt", FILTERS, ids=lambda f: f.name)
def test_online_stepper_matches_whole_trial_run(filt, rng):
    """Frame-by-frame use (live demo) gives exactly the same numbers as the analysis."""
    t, xy, measured = _noisy_reach(rng)
    batch = filt.run(t, xy, measured)
    stepper = filt.new_state()
    for i in range(len(t)):
        p, v, a, used = stepper.update(t[i], xy[i], measured[i])
        np.testing.assert_array_equal(p, batch.position[i])
        np.testing.assert_array_equal(v, batch.velocity[i])
        assert used == batch.updated[i]


def test_polyfit_is_exact_on_quadratic_with_uneven_frame_times(rng):
    t = np.cumsum(rng.uniform(0.010, 0.025, 60))
    a = np.array([30.0, -12.0])
    v0 = np.array([5.0, 2.0])
    xy = 1.0 + v0 * t[:, None] + 0.5 * a * t[:, None] ** 2
    out = CausalPolyfitFilter(window=7, order=2).run(t, xy, np.ones(len(t), bool))
    np.testing.assert_allclose(out.position[10:], xy[10:], atol=1e-8)
    np.testing.assert_allclose(out.velocity[10:], v0 + a * t[10:, None], atol=1e-6)
    np.testing.assert_allclose(out.acceleration[10:], np.broadcast_to(a, (50, 2)), atol=1e-5)


def test_kalman_tracks_constant_acceleration():
    t = np.arange(0, 2.0, 1 / 60)
    a = np.array([20.0, -10.0])
    xy = 0.5 * a * t[:, None] ** 2
    out = KalmanCAFilter(jerk_psd=1e3, measurement_std=0.01).run(t, xy, np.ones(len(t), bool))
    # After a short convergence period the estimates match the true motion.
    np.testing.assert_allclose(out.velocity[-1], a * t[-1], rtol=1e-3)
    np.testing.assert_allclose(out.acceleration[-1], a, rtol=1e-2)


def test_kalman_predicts_through_missing_frames_and_waits_for_first_detection():
    t = np.arange(10) / 60
    xy = np.column_stack([np.arange(10.0), np.zeros(10)])
    measured = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1], bool)
    out = KalmanCAFilter(jerk_psd=1e4, measurement_std=0.1).run(t, xy, measured)
    assert np.all(np.isnan(out.position[:2]))  # nothing known before the first detection
    assert np.all(np.isfinite(out.position[2:]))
    np.testing.assert_array_equal(out.updated, measured)


def test_default_filter_does_not_report_false_movement_when_still(cfg, rng):
    """With the default settings, detector jitter on a still hand must stay below the onset threshold."""
    filt = make_filter(cfg)
    t = np.arange(0, 3.0, 1 / cfg["camera"]["fps"])
    noise_cm = cfg["kinematics"]["kalman_ca"]["measurement_std_cm"]
    seg = cfg["kinematics"]["segmentation"]
    for _ in range(20):
        xy = np.array([10.0, 20.0]) + rng.normal(0, noise_cm, (len(t), 2))
        out = filt.run(t, xy, np.ones(len(t), bool))
        assert find_onset(out.speed, seg["onset_speed_cm_s"], seg["onset_min_frames"]) is None


def test_fill_short_gaps():
    t = np.arange(20, dtype=float)
    xy = np.column_stack([t * 2.0, t * -1.0])
    detected = np.ones(20, bool)
    detected[0] = False  # leading gap: not filled
    detected[3:6] = False  # 3 missing frames: filled
    detected[9:13] = False  # 4 missing frames: too long, stays NaN
    detected[19] = False  # trailing gap: not filled
    filled, interp = fill_short_gaps(t, xy, detected, max_gap_frames=3)
    np.testing.assert_allclose(filled[3:6], xy[3:6])
    assert interp[3:6].all() and interp.sum() == 3
    assert np.isnan(filled[[0, 9, 10, 11, 12, 19]]).all()


def test_segmentation_with_hysteresis(rng):
    t = np.arange(0, 3, 1 / 60)
    speed = 60 * np.exp(-0.5 * ((t - 1.5) / 0.15) ** 2)  # bell-shaped speed profile
    speed += rng.normal(0, 0.3, len(t))
    seg = segment_movement(speed, onset_speed=5, offset_speed=3, onset_min_frames=3, offset_min_frames=6)
    assert seg.found and seg.offset_found
    assert t[seg.onset_index] == pytest.approx(1.5 - 0.15 * np.sqrt(2 * np.log(60 / 5)), abs=0.05)
    assert t[seg.offset_index] == pytest.approx(1.5 + 0.15 * np.sqrt(2 * np.log(60 / 3)), abs=0.05)
    assert t[seg.peak_index] == pytest.approx(1.5, abs=0.03)


def test_segmentation_edge_cases():
    still = np.full(100, 1.0)
    assert not segment_movement(still, 5, 3, 3, 6).found
    rising = np.linspace(0, 50, 100)  # still moving at the end of the recording
    seg = segment_movement(rising, 5, 3, 3, 6)
    assert seg.found and not seg.offset_found and seg.offset_index == 99


def test_ground_truth_interpolation():
    t = np.array([0.0, 0.1, 0.2, 0.4])
    xy = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])
    q = np.array([0.0, 0.05, 0.1, 0.3, 0.45, -0.1])
    out, valid = ground_truth_at(t, xy, q, max_gap_s=0.15)
    np.testing.assert_allclose(out[0], [0, 0])
    np.testing.assert_allclose(out[1], [0.5, 1.0])
    np.testing.assert_allclose(out[2], [1, 2])
    # 0.3 lies in a 0.2 s gap (> 0.15 s): no trustworthy truth there.
    assert valid.tolist() == [True, True, True, False, False, False]
    out2, valid2 = ground_truth_at(t, xy, np.array([0.3]), max_gap_s=0.25)
    assert valid2[0]
    np.testing.assert_allclose(out2[0], [3.0, 6.0])
