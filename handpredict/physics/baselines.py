"""Simple physics baselines: stay still, constant velocity, constant acceleration.

These are the "ablations": they show how much of the minimum-jerk model's
accuracy comes from just knowing the current velocity (or acceleration).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..predictor import Prediction, Predictor
from ..trial import History


class ZeroMotionPredictor(Predictor):
    """"Stay where you are": predicts the latest measured position for every horizon.

    Any useful model must beat this; it is the sanity floor of the study.
    """

    name = "zero_motion"
    family = "baseline"

    def predict(self, history: History, horizons_s: np.ndarray) -> Prediction:
        return Prediction(np.tile(history.current_xy, (len(horizons_s), 1)))


def latest_state(history: History) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position, velocity and acceleration from the causal filter at the current frame."""
    return history.filt_xy[-1], history.filt_v[-1], history.filt_a[-1]


class ConstantVelocityPredictor(Predictor):
    """x(t + h) = x + v * h, with x and v from the causal filter."""

    name = "constant_velocity"

    def predict(self, history: History, horizons_s: np.ndarray) -> Prediction:
        return Prediction(constant_velocity(history, horizons_s))


def constant_velocity(history: History, horizons_s: np.ndarray) -> np.ndarray:
    x, v, _ = latest_state(history)
    return x + np.outer(horizons_s, v)


def euler_roll(
    x0: np.ndarray,
    v0: np.ndarray,
    accel_at: Callable[[np.ndarray], np.ndarray],
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Step the state forward with the Newton-Euler (forward Euler) update.

        v[n+1] = v[n] + a[n] * dt
        x[n+1] = x[n] + v[n] * dt

    ``accel_at(times)`` gives the acceleration (shape (n, 2)) at the step
    times 0, dt, 2 dt, ... Because a[n] does not depend on x or v, the loop
    collapses into two running sums, which ``np.cumsum`` computes without a
    slow Python loop. ``tests/test_physics.py`` checks that this gives exactly
    the same numbers as the plain loop. Returns positions at steps 0..n_steps.
    """
    times = np.arange(n_steps) * dt
    accel = np.asarray(accel_at(times), dtype=float).reshape(n_steps, 2)
    velocity = np.empty((n_steps + 1, 2))
    velocity[0] = v0
    velocity[1:] = v0 + dt * np.cumsum(accel, axis=0)
    position = np.empty((n_steps + 1, 2))
    position[0] = x0
    position[1:] = x0 + dt * np.cumsum(velocity[:-1], axis=0)
    return position


def euler_roll_loop(x0, v0, accel_at, dt: float, n_steps: int) -> np.ndarray:
    """Plain-loop version of ``euler_roll`` (slow; kept to show the maths is identical)."""
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    out = [x.copy()]
    for n in range(n_steps):
        a = np.asarray(accel_at(np.array([n * dt])), dtype=float).reshape(2)
        x, v = x + v * dt, v + a * dt
        out.append(x.copy())
    return np.array(out)


def horizon_steps(horizons_s: np.ndarray, dt: float) -> np.ndarray:
    """Step index for each horizon; the internal dt must divide every horizon exactly."""
    steps = np.rint(np.asarray(horizons_s) / dt).astype(int)
    if not np.allclose(steps * dt, horizons_s, atol=1e-9):
        raise ValueError(f"internal step {dt * 1000} ms does not divide the horizons {horizons_s * 1000} ms")
    return steps


class EulerConstantAccelerationPredictor(Predictor):
    """The proposal's Newton-Euler model: roll x, v forward with constant acceleration a.

    x, v and a come from the causal filter at the current frame. Optionally
    (``decay_tau_s``) the acceleration fades exponentially, because a real
    hand cannot keep accelerating; the default is the proposal's constant a.
    """

    name = "euler_constant_acceleration"

    def __init__(self, dt_s: float = 0.001, decay_tau_s: float | None = None) -> None:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        self.dt = float(dt_s)
        self.decay_tau = None if decay_tau_s in (None, 0) else float(decay_tau_s)

    def predict(self, history: History, horizons_s: np.ndarray) -> Prediction:
        x, v, a = latest_state(history)
        steps = horizon_steps(horizons_s, self.dt)

        def accel_at(times: np.ndarray) -> np.ndarray:
            if self.decay_tau is None:
                return np.tile(a, (len(times), 1))
            return np.exp(-times / self.decay_tau)[:, None] * a

        path = euler_roll(x, v, accel_at, self.dt, int(steps.max()))
        return Prediction(path[steps])

    def describe(self) -> dict:
        return {**super().describe(), "dt_s": self.dt, "decay_tau_s": self.decay_tau}
