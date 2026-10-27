import numpy as np

from handpredict.synthetic import ReachSpec, frame_times, minimum_jerk_profile, reach_position, sample_reach_spec, trial_plan


def test_minimum_jerk_profile_boundary_conditions():
    tau = np.linspace(0, 1, 101)
    s = minimum_jerk_profile(tau)
    assert s[0] == 0 and s[-1] == 1
    assert np.all(np.diff(s) >= 0)
    assert minimum_jerk_profile(np.array([0.5]))[0] == 0.5  # symmetric profile


def test_reach_starts_and_ends_where_specified():
    spec = ReachSpec(
        start_px=np.array([100.0, 200.0]),
        target_px=np.array([400.0, 200.0]),
        endpoint_px=np.array([405.0, 195.0]),
        onset_s=0.5,
        duration_s=0.8,
        curvature=0.05,
        asymmetry=0.2,
    )
    t = np.array([0.0, 0.5, 1.3, 2.0])
    p = reach_position(spec, t)
    np.testing.assert_allclose(p[0], spec.start_px)
    np.testing.assert_allclose(p[1], spec.start_px)
    np.testing.assert_allclose(p[2], spec.endpoint_px, atol=1e-9)
    np.testing.assert_allclose(p[3], spec.endpoint_px, atol=1e-9)
    mid = reach_position(spec, np.array([0.9]))[0]
    # the path bows away from the straight line in the middle of the movement
    assert abs(mid[1] - 200.0) > 1.0


def test_frame_times_are_strictly_increasing(rng):
    t = frame_times(rng, fps=60, total_s=3.0, jitter_ms=0.5, drop_probability=0.05)
    assert t[0] == 0.0
    assert np.all(np.diff(t) > 0)
    assert 150 < len(t) <= 181


def test_sample_spec_and_plan(cfg, rng):
    spec = sample_reach_spec(rng, cfg["synthetic"], "T2", "fast", go_time_s=0.5)
    lo, hi = cfg["synthetic"]["durations_s"]["fast"]
    assert lo <= spec.duration_s <= hi
    assert spec.onset_s > 0.5
    plan = trial_plan(["slow", "fast"], ["T1", "T2", "T3"], 2, rng)
    assert len(plan) == 12 and sorted(set(plan)) == sorted({(s, t) for s in ["slow", "fast"] for t in ["T1", "T2", "T3"]})
