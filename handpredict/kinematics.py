"""Kinematics: from noisy measured positions to position, velocity and acceleration.

The most important rule in this file is **causality**. A predictor working at
time t may only use samples taken at or before t. A normal "smoothing" filter
that looks at samples on both sides of t (for example ``scipy.signal.filtfilt``
or a centred Savitzky-Golay filter) would secretly use the future, which makes
predictions look better than they could ever be in real time. That is a data
leak, so every filter here walks through the samples in time order and its
output at sample i depends only on samples 0..i. ``tests/test_kinematics.py``
checks this by changing a future sample and asserting that no earlier output
changes.

Two causal filters are provided:

* ``KalmanCAFilter``: a Kalman filter with a constant-acceleration motion
  model. It keeps a running estimate of position, velocity and acceleration
  and, at each new frame, blends "where physics says the hand should be now"
  with "where the camera says it is", weighted by how much it trusts each.
* ``CausalPolyfitFilter``: fits a parabola to the most recent samples and
  reads off position (value), velocity (slope) and acceleration (curvature)
  at the newest sample. It is a causal version of the Savitzky-Golay filter.

Both use the real time between frames (frames are not perfectly evenly
spaced), so velocities are in cm/s and accelerations in cm/s^2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class KinematicState:
    """Filter output for every sample of a trial (arrays of shape (n, 2) and (n,))."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    updated: np.ndarray  # True where a real measurement was used at that sample

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity, axis=1)


class FilterStepper(Protocol):
    """Online form of a causal filter: feed one sample at a time, get the current estimate."""

    def update(self, t: float, z: np.ndarray, measured: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Returns (position, velocity, acceleration, used_measurement); NaN before the first detection."""
        ...


class CausalFilter(Protocol):
    """A filter that can run online (``new_state``) or over a whole trial (``run``)."""

    name: str

    def new_state(self) -> FilterStepper:
        ...

    def run(self, t: np.ndarray, z: np.ndarray, measured: np.ndarray) -> KinematicState:
        ...


def run_stepper(filt: CausalFilter, t: np.ndarray, z: np.ndarray, measured: np.ndarray) -> KinematicState:
    """Run a filter over a whole trial by feeding samples one by one, in time order.

    Analysis (whole trials) and the live demo (one frame at a time) both go
    through ``FilterStepper.update``, so they give exactly the same numbers,
    and the output at sample i can only depend on samples 0..i.
    """
    t, z, measured = _check_inputs(t, z, measured)
    n = len(t)
    pos = np.full((n, 2), np.nan)
    vel = np.full((n, 2), np.nan)
    acc = np.full((n, 2), np.nan)
    updated = np.zeros(n, dtype=bool)
    stepper = filt.new_state()
    for i in range(n):
        pos[i], vel[i], acc[i], updated[i] = stepper.update(float(t[i]), z[i], bool(measured[i]))
    return KinematicState(pos, vel, acc, updated)


_NAN2 = np.full(2, np.nan)


# -----------------------------------------------------------------------------
# Kalman filter with a constant-acceleration model
# -----------------------------------------------------------------------------
class KalmanCAFilter:
    """Constant-acceleration Kalman filter, run independently on x and y.

    State per axis: [position, velocity, acceleration]. Between frames the
    state is pushed forward with the equations of motion

        p' = p + v*dt + a*dt^2/2,   v' = v + a*dt,   a' = a,

    plus "process noise" that allows the acceleration to drift, because a real
    hand changes its acceleration (it has jerk). ``jerk_psd`` sets how large
    those changes may be. At each frame with a detection the estimate is
    corrected towards the measurement; frames without a detection only get the
    prediction step.
    """

    name = "kalman_ca"

    def __init__(
        self,
        jerk_psd: float,
        measurement_std: float,
        initial_velocity_std: float = 10.0,
        initial_acceleration_std: float = 100.0,
    ) -> None:
        if jerk_psd <= 0 or measurement_std <= 0:
            raise ValueError("jerk_psd and measurement_std must be positive")
        self.q = float(jerk_psd)
        self.r = float(measurement_std) ** 2
        self.p0 = np.diag([self.r, initial_velocity_std**2, initial_acceleration_std**2])

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        return np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])

    def _process_noise(self, dt: float) -> np.ndarray:
        # Continuous white-noise-jerk model integrated over one time step.
        d2, d3, d4, d5 = dt**2, dt**3, dt**4, dt**5
        return self.q * np.array(
            [[d5 / 20.0, d4 / 8.0, d3 / 6.0], [d4 / 8.0, d3 / 3.0, d2 / 2.0], [d3 / 6.0, d2 / 2.0, dt]]
        )

    def new_state(self) -> "_KalmanStepper":
        return _KalmanStepper(self)

    def run(self, t: np.ndarray, z: np.ndarray, measured: np.ndarray) -> KinematicState:
        return run_stepper(self, t, z, measured)


class _KalmanStepper:
    """The running estimate of one Kalman filter (one per trial / live session)."""

    def __init__(self, params: KalmanCAFilter) -> None:
        self.p = params
        self.state: np.ndarray | None = None  # shape (3, 2): rows = p, v, a; columns = x, y
        self.cov: np.ndarray | None = None  # 3x3, identical for both axes (same model and noise)
        self.t_prev = 0.0

    def update(self, t: float, z: np.ndarray, measured: bool):
        measured = bool(measured) and bool(np.all(np.isfinite(z)))
        used = False
        if self.state is None:
            if not measured:
                return _NAN2, _NAN2, _NAN2, False  # nothing is known before the first detection
            self.state = np.zeros((3, 2))
            self.state[0] = z
            self.cov = self.p.p0.copy()
            used = True
        else:
            dt = float(t - self.t_prev)
            if dt < 0:
                raise ValueError("timestamps must be increasing")
            f = self.p._transition(dt)
            self.state = f @ self.state
            self.cov = f @ self.cov @ f.T + self.p._process_noise(dt)
            if measured:
                # Kalman gain: how far to move towards the measurement.
                innovation_var = self.cov[0, 0] + self.p.r
                gain = self.cov[:, 0] / innovation_var
                self.state = self.state + np.outer(gain, z - self.state[0])
                self.cov = self.cov - np.outer(gain, self.cov[0, :])
                self.cov = 0.5 * (self.cov + self.cov.T)
                used = True
        self.t_prev = float(t)
        return self.state[0].copy(), self.state[1].copy(), self.state[2].copy(), used


# -----------------------------------------------------------------------------
# Causal local polynomial fit (causal Savitzky-Golay)
# -----------------------------------------------------------------------------
class CausalPolyfitFilter:
    """Fit a polynomial to the last ``window`` measured samples, evaluated at the newest time."""

    name = "causal_polyfit"

    def __init__(self, window: int = 9, order: int = 2) -> None:
        if window < 2 or order < 0:
            raise ValueError("window must be >= 2 and order >= 0")
        self.window = int(window)
        self.order = int(order)

    def new_state(self) -> "_PolyfitStepper":
        return _PolyfitStepper(self)

    def run(self, t: np.ndarray, z: np.ndarray, measured: np.ndarray) -> KinematicState:
        return run_stepper(self, t, z, measured)


class _PolyfitStepper:
    def __init__(self, params: CausalPolyfitFilter) -> None:
        self.p = params
        self.recent_t: list[float] = []  # the most recent measured samples (all in the past)
        self.recent_z: list[np.ndarray] = []

    def update(self, t: float, z: np.ndarray, measured: bool):
        measured = bool(measured) and bool(np.all(np.isfinite(z)))
        if measured:
            self.recent_t.append(float(t))
            self.recent_z.append(np.array(z, dtype=float))
            if len(self.recent_t) > self.p.window:
                self.recent_t.pop(0)
                self.recent_z.pop(0)
        if not self.recent_t:
            return _NAN2, _NAN2, _NAN2, False
        order = min(self.p.order, len(self.recent_t) - 1)
        # Time relative to "now", so the polynomial's coefficients are the
        # position, velocity and acceleration/2 at the current time.
        tau = np.asarray(self.recent_t) - t
        design = np.vander(tau, order + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(design, np.asarray(self.recent_z), rcond=None)
        vel = coef[1] if order >= 1 else np.zeros(2)
        acc = 2.0 * coef[2] if order >= 2 else np.zeros(2)
        return coef[0].copy(), np.array(vel), np.array(acc), measured


def make_filter(cfg: Mapping) -> CausalFilter:
    """Build the causal filter named in ``cfg['kinematics']['filter']``."""
    kin = cfg["kinematics"]
    if kin["filter"] == "kalman_ca":
        k = kin["kalman_ca"]
        return KalmanCAFilter(
            jerk_psd=k["jerk_psd_cm2_s5"],
            measurement_std=k["measurement_std_cm"],
            initial_velocity_std=k["initial_velocity_std_cm_s"],
            initial_acceleration_std=k["initial_acceleration_std_cm_s2"],
        )
    if kin["filter"] == "causal_polyfit":
        p = kin["causal_polyfit"]
        return CausalPolyfitFilter(window=p["window"], order=p["order"])
    raise ValueError(f"Unknown filter {kin['filter']!r}")


# -----------------------------------------------------------------------------
# Missing detections
# -----------------------------------------------------------------------------
def fill_short_gaps(
    t: np.ndarray, xy: np.ndarray, detected: np.ndarray, max_gap_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate gaps of at most ``max_gap_frames`` missing frames.

    Returns the filled positions and a boolean ``interpolated`` flag. Longer
    gaps, and gaps at the very start or end, stay NaN. Interpolated points are
    never used as prediction times or as ground truth: an interpolated value
    at frame k depends on the next real detection after k, so it only becomes
    known once that later frame has arrived.
    """
    t = np.asarray(t, dtype=float)
    out = np.array(xy, dtype=float, copy=True)
    detected = np.asarray(detected, dtype=bool)
    out[~detected] = np.nan
    interpolated = np.zeros(len(t), dtype=bool)
    i = 0
    n = len(t)
    while i < n:
        if detected[i]:
            i += 1
            continue
        start = i
        while i < n and not detected[i]:
            i += 1
        end = i  # first detected index after the gap (or n)
        gap = end - start
        if start == 0 or end == n or gap > max_gap_frames:
            continue
        left, right = start - 1, end
        w = (t[start:end] - t[left]) / (t[right] - t[left])
        out[start:end] = out[left] + w[:, None] * (out[right] - out[left])
        interpolated[start:end] = True
    return out, interpolated


# -----------------------------------------------------------------------------
# Movement segmentation (onset / offset)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Segmentation:
    """Where the movement starts and ends inside a trial (sample indices)."""

    onset_index: int | None
    offset_index: int | None
    peak_index: int | None
    offset_found: bool  # False if the hand was still moving when the recording ended

    @property
    def found(self) -> bool:
        return self.onset_index is not None


def find_run(values_above: np.ndarray, min_frames: int, start: int = 0) -> int | None:
    """First index i >= start where ``values_above[i : i + min_frames]`` are all True."""
    run = 0
    for i in range(start, len(values_above)):
        run = run + 1 if values_above[i] else 0
        if run >= min_frames:
            return i - min_frames + 1
    return None


def find_onset(speed: np.ndarray, onset_speed: float, min_frames: int) -> int | None:
    """Index where the speed first stays above ``onset_speed`` for ``min_frames`` frames.

    Only past samples are needed, so predictors can call this on their own
    history to detect the onset online.
    """
    above = np.nan_to_num(speed, nan=0.0) > onset_speed
    return find_run(above, min_frames)


def segment_movement(
    speed: np.ndarray,
    onset_speed: float,
    offset_speed: float,
    onset_min_frames: int,
    offset_min_frames: int,
) -> Segmentation:
    """Speed-threshold segmentation with hysteresis.

    The movement starts when the speed stays above ``onset_speed`` and ends
    when it stays below the lower ``offset_speed``. Using two thresholds
    (hysteresis) stops a noisy speed signal near one threshold from switching
    "moving / not moving" back and forth.
    """
    speed = np.asarray(speed, dtype=float)
    onset = find_onset(speed, onset_speed, onset_min_frames)
    if onset is None:
        return Segmentation(None, None, None, False)
    clean = np.nan_to_num(speed, nan=0.0)
    below = clean < offset_speed
    # The offset is searched only after the movement has clearly started.
    offset_start = find_run(below, offset_min_frames, start=onset + onset_min_frames)
    if offset_start is None:
        offset = len(speed) - 1
        found = False
    else:
        offset = offset_start
        found = True
    peak = onset + int(np.argmax(clean[onset : offset + 1]))
    return Segmentation(onset, offset, peak, found)


def segment_from_config(speed: np.ndarray, cfg: Mapping) -> Segmentation:
    seg = cfg["kinematics"]["segmentation"]
    return segment_movement(
        speed,
        onset_speed=seg["onset_speed_cm_s"],
        offset_speed=seg["offset_speed_cm_s"],
        onset_min_frames=seg["onset_min_frames"],
        offset_min_frames=seg["offset_min_frames"],
    )


# -----------------------------------------------------------------------------
# Ground truth
# -----------------------------------------------------------------------------
def ground_truth_at(
    t_meas: np.ndarray, xy_meas: np.ndarray, t_query: np.ndarray, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Measured hand position at arbitrary times, by linear interpolation.

    ``t_meas`` / ``xy_meas`` must contain only REAL measurements (no filled
    gaps, no filtering): the truth is what the camera actually saw. A query is
    valid only if it lies inside the measured time range and its two
    neighbouring measurements are at most ``max_gap_s`` apart. Because the raw
    measurement still contains detector noise, even a perfect predictor would
    show a small error; that noise floor is the same for every model.
    """
    t_meas = np.asarray(t_meas, dtype=float)
    xy_meas = np.asarray(xy_meas, dtype=float)
    t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
    out = np.full((len(t_query), 2), np.nan)
    valid = np.zeros(len(t_query), dtype=bool)
    if len(t_meas) == 0:
        return out, valid
    if np.any(np.diff(t_meas) <= 0):
        raise ValueError("measurement times must be strictly increasing")
    eps = 1e-9
    right = np.searchsorted(t_meas, t_query, side="left")
    for q, (tq, r) in enumerate(zip(t_query, right)):
        if r < len(t_meas) and abs(t_meas[r] - tq) <= eps:
            out[q] = xy_meas[r]
            valid[q] = True
            continue
        if r == 0 or r >= len(t_meas):
            continue  # before the first or after the last measurement
        left = r - 1
        if t_meas[r] - t_meas[left] > max_gap_s + eps:
            continue  # inside a long gap: no trustworthy ground truth
        w = (tq - t_meas[left]) / (t_meas[r] - t_meas[left])
        out[q] = xy_meas[left] + w * (xy_meas[r] - xy_meas[left])
        valid[q] = True
    return out, valid


def _check_inputs(t: np.ndarray, z: np.ndarray, measured: np.ndarray):
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    measured = np.asarray(measured, dtype=bool) & np.all(np.isfinite(z), axis=1)
    if z.ndim != 2 or z.shape[1] != 2 or len(z) != len(t) or len(measured) != len(t):
        raise ValueError("expected t (n,), z (n, 2) and measured (n,)")
    return t, z, measured
