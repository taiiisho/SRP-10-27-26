"""Tests for the physics-based predictors."""

import numpy as np
import pytest

from handpredict.kinematics import make_filter
from handpredict.physics import make_physics_predictors
from handpredict.physics.baselines import (
    ConstantVelocityPredictor,
    EulerConstantAccelerationPredictor,
    ZeroMotionPredictor,
    euler_roll,
    euler_roll_loop,
)
from handpredict.physics.minimum_jerk import (
    MinJerkFit,
    mj_dds,
    mj_ds,
    mj_s,
    phi_v,
    phi_v_dd,
    predict_closed_form,
    predict_euler_rolled,
    predict_replan_closed,
    predict_replan_euler,
    replan_terms,
)
from handpredict.synthetic import ReachSpec, make_trial_data
from handpredict.trial import History, TrialContext, make_history, process_trial

HORIZONS = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])


def _context(target=None):
    return TrialContext("t1", "s1", "p1", "fast", "T1", None, target, "cm", 60.0)


def _history_from_state(x, v, a):
    """A one-sample History whose causal-filter state is exactly (x, v, a)."""
    x, v, a = (np.atleast_2d(np.asarray(q, float)) for q in (x, v, a))
    return History(0, np.array([1.0]), x.copy(), np.array([True]), x, v, a, _context())


# -- minimum-jerk profile ------------------------------------------------------
def test_minimum_jerk_profile_boundaries_and_derivatives():
    assert mj_s(np.array([0.0]))[0] == 0.0 and mj_s(np.array([1.0]))[0] == 1.0
    assert mj_s(np.array([0.5]))[0] == pytest.approx(0.5)
    for f in (mj_ds, mj_dds):
        assert f(np.array([0.0]))[0] == 0.0 and f(np.array([1.0]))[0] == pytest.approx(0.0)
    tau = np.linspace(0.01, 0.99, 99)
    h = 1e-6
    np.testing.assert_allclose((mj_s(tau + h) - mj_s(tau - h)) / (2 * h), mj_ds(tau), atol=1e-5)
    np.testing.assert_allclose((mj_ds(tau + h) - mj_ds(tau - h)) / (2 * h), mj_dds(tau), atol=1e-4)
    assert mj_ds(np.array([0.5]))[0] == pytest.approx(1.875)  # peak speed = 1.875 * distance / T


# -- Newton-Euler stepping ------------------------------------------------------
def test_vectorised_euler_equals_plain_loop(rng):
    a0 = rng.normal(0, 50, 2)
    accel = lambda times: np.outer(np.cos(7 * times), a0)  # noqa: E731
    fast = euler_roll(np.array([1.0, 2.0]), np.array([3.0, -4.0]), accel, 0.001, 400)
    slow = euler_roll_loop(np.array([1.0, 2.0]), np.array([3.0, -4.0]), accel, 0.001, 400)
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-10)


def _example_fit():
    return MinJerkFit(
        ok=True, t0=0.80, duration=0.6, x0=np.array([10.0, 30.0]), xf=np.array([45.0, 12.0])
    )


def test_closed_form_and_euler_rolled_minimum_jerk_agree():
    """Rolling the minimum-jerk acceleration forward with small Euler steps reproduces the polynomial."""
    fit = _example_fit()
    distance = float(np.linalg.norm(fit.xf - fit.x0))  # 39 cm
    worst = 0.0
    for t_now in np.linspace(0.7, 1.5, 33):  # before, during and after the movement
        closed = predict_closed_form(fit, t_now, HORIZONS)
        rolled = predict_euler_rolled(fit, t_now, HORIZONS, dt=1e-4)  # the configured 0.1 ms step
        worst = max(worst, float(np.abs(closed - rolled).max()))
    # Forward Euler's error is first order in dt (see the next test). With a
    # 0.1 ms step it stays below 0.1 % of the movement distance (< 0.4 mm
    # here), far below the ~1.5 mm detector noise.
    assert worst < 1e-3 * distance


def test_euler_error_shrinks_linearly_with_step():
    """Forward Euler is first-order accurate: halving dt roughly halves the error."""
    fit = _example_fit()
    errors = []
    for dt in (1e-3, 5e-4, 2.5e-4):
        diff = predict_euler_rolled(fit, 0.9, HORIZONS, dt) - predict_closed_form(fit, 0.9, HORIZONS)
        errors.append(np.abs(diff).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.25)


def test_replan_basis_function_boundary_conditions():
    h = 1e-6
    d = lambda f, u: (f(u + h) - f(u - h)) / (2 * h)  # noqa: E731
    assert phi_v(np.array([0.0]))[0] == 0 and phi_v(np.array([1.0]))[0] == pytest.approx(0, abs=1e-12)
    assert d(phi_v, np.array([1e-3]))[0] == pytest.approx(1.0, abs=1e-4)  # unit slope at the start
    assert d(phi_v, np.array([1 - 1e-3]))[0] == pytest.approx(0.0, abs=1e-4)  # at rest at the end
    u = np.linspace(0.01, 0.99, 50)
    numeric_dd = (phi_v(u + 1e-4) - 2 * phi_v(u) + phi_v(u - 1e-4)) / 1e-8
    np.testing.assert_allclose(numeric_dd, phi_v_dd(u), atol=1e-3)


def test_replan_closed_and_euler_agree_and_start_at_measured_state():
    fit = _example_fit()
    distance = float(np.linalg.norm(fit.xf - fit.x0))
    for t_now in (0.9, 1.1, 1.3, 1.5):
        x_meas = fit.position(np.array([t_now]))[0] + np.array([0.8, -1.2])
        v_meas = fit.velocity(np.array([t_now]))[0] + np.array([5.0, 3.0])
        terms = replan_terms(fit, t_now, x_meas, v_meas, min_remaining=0.1)
        closed = predict_replan_closed(fit, terms, t_now, HORIZONS)
        rolled = predict_replan_euler(fit, terms, t_now, HORIZONS, dt=1e-4)
        assert np.abs(closed - rolled).max() < 1e-3 * distance
        start = predict_replan_closed(fit, terms, t_now, np.array([0.0]))[0]
        np.testing.assert_allclose(start, x_meas, atol=1e-12)  # re-planning starts where the hand IS
    # With no mismatch, re-planning is the ordinary minimum-jerk prediction.
    t_now = 1.0
    terms = replan_terms(fit, t_now, fit.position(np.array([t_now]))[0], fit.velocity(np.array([t_now]))[0], 0.1)
    np.testing.assert_allclose(
        predict_replan_closed(fit, terms, t_now, HORIZONS), predict_closed_form(fit, t_now, HORIZONS), atol=1e-12
    )


# -- baselines -------------------------------------------------------------------
def test_constant_velocity_is_exact_on_constant_velocity_motion():
    x, v = np.array([5.0, -2.0]), np.array([40.0, 15.0])
    pred = ConstantVelocityPredictor().predict(_history_from_state(x, v, [0, 0]), HORIZONS)
    np.testing.assert_allclose(pred.positions, x + np.outer(HORIZONS, v), atol=1e-12)


def test_euler_constant_acceleration_matches_kinematics_equation():
    x, v, a = np.array([1.0, 2.0]), np.array([30.0, -10.0]), np.array([200.0, 50.0])
    dt = 0.001
    pred = EulerConstantAccelerationPredictor(dt_s=dt).predict(_history_from_state(x, v, a), HORIZONS)
    exact = x + np.outer(HORIZONS, v) + 0.5 * np.outer(HORIZONS**2, a)
    # Forward Euler under-shoots x = x0 + v t + a t^2 / 2 by exactly a * t * dt / 2.
    np.testing.assert_allclose(exact - pred.positions, 0.5 * np.outer(HORIZONS, a) * dt, atol=1e-9)
    decaying = EulerConstantAccelerationPredictor(dt_s=dt, decay_tau_s=0.1).predict(
        _history_from_state(x, v, a), HORIZONS
    )
    assert np.linalg.norm(decaying.positions[-1] - x) < np.linalg.norm(pred.positions[-1] - x)


def test_zero_motion_predicts_current_position():
    h = _history_from_state([3.0, 4.0], [100.0, 0.0], [0.0, 0.0])
    np.testing.assert_array_equal(ZeroMotionPredictor().predict(h, HORIZONS).positions, np.tile([3.0, 4.0], (7, 1)))


# -- minimum-jerk predictor on whole trials ------------------------------------------
def _perfect_trial(cfg, rng, endpoint_offset_px=(0.0, 0.0), duration=0.8):
    spec = ReachSpec(
        start_px=np.array([330.0, 360.0]),
        target_px=np.array([800.0, 300.0]),
        endpoint_px=np.array([800.0, 300.0]) + np.asarray(endpoint_offset_px),
        onset_s=0.9,
        duration_s=duration,
        curvature=0.0,
        asymmetry=0.0,
    )
    data = make_trial_data(spec, cfg, rng, "perfect", "medium", "T1", timing=False)
    return process_trial(data, make_filter(cfg), cfg), spec


def _truth(spec, cfg, t):
    from handpredict.synthetic import reach_position

    return reach_position(spec, t) * cfg["synthetic"]["cm_per_px"]


def test_minimum_jerk_recovers_noiseless_trajectory(cfg, rng):
    """On a perfect minimum-jerk movement with the right target, the fitted model is (nearly) exact."""
    trial, spec = _perfect_trial(cfg, rng)
    models = make_physics_predictors(cfg)
    used, worst = 0, 0.0
    for j in range(trial.data.n):
        h = make_history(trial, j)
        for name in ("minimum_jerk_euler", "minimum_jerk_closed"):
            pred = models[name].predict(h, HORIZONS)
            if pred.fallback is None:
                used += 1
                truth = _truth(spec, cfg, h.now + HORIZONS)
                worst = max(worst, float(np.linalg.norm(pred.positions - truth, axis=1).max()))
    assert used > 2 * 40  # the fit is accepted for most of the movement
    assert worst < 0.05  # cm


def test_minimum_jerk_falls_back_before_onset_and_holds_at_target(cfg, rng):
    trial, spec = _perfect_trial(cfg, rng)
    model = make_physics_predictors(cfg)["minimum_jerk_euler"]
    early = make_history(trial, 20)  # hand still resting
    pred = model.predict(early, HORIZONS)
    assert pred.fallback == "no_onset_yet"
    np.testing.assert_allclose(pred.positions, ConstantVelocityPredictor().predict(early, HORIZONS).positions)
    late = make_history(trial, trial.data.n - 1)  # long after the movement ended
    pred = model.predict(late, HORIZONS)
    assert pred.fallback is None
    np.testing.assert_allclose(pred.positions, np.tile(trial.context.target_xy, (7, 1)), atol=1e-6)


def test_free_target_ablation_finds_the_real_end_point(cfg, rng):
    """If the hand stops 3 cm away from the marker, only the free-target fit can know that."""
    offset_px = np.array([0.0, 3.0 / cfg["synthetic"]["cm_per_px"]])
    trial, spec = _perfect_trial(cfg, rng, endpoint_offset_px=offset_px)
    models = make_physics_predictors(cfg)
    j = int(np.searchsorted(trial.data.t, spec.onset_s + 0.7 * spec.duration_s))
    h = make_history(trial, j)
    free = models["minimum_jerk_free_target"].fitter.fit(h)
    assert free.ok
    np.testing.assert_allclose(free.xf, spec.endpoint_px * cfg["synthetic"]["cm_per_px"], atol=0.05)
    truth_end = _truth(spec, cfg, np.array([h.now + 0.5]))[0]
    err_free = np.linalg.norm(models["minimum_jerk_free_target"].predict(h, HORIZONS).positions[-1] - truth_end)
    err_known = np.linalg.norm(models["minimum_jerk_euler"].predict(h, HORIZONS).positions[-1] - truth_end)
    assert err_free < 0.1 < err_known


def test_physics_registry_builds_configured_models(cfg):
    models = make_physics_predictors(cfg)
    assert list(models) == cfg["physics"]["models"]
    assert models["minimum_jerk_euler"].fitter is models["minimum_jerk_closed"].fitter
    assert cfg["physics"]["headline"] in models
