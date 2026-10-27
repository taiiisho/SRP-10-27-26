"""The minimum-jerk model of reaching (Flash & Hogan, 1985): the headline physics model.

Flash and Hogan found that smooth point-to-point arm movements are well
described by the path that minimises the total squared *jerk* (the rate of
change of acceleration). Solving that optimisation (Euler-Lagrange equation)
gives a fifth-order polynomial:

    x(t) = x0 + (xf - x0) * s(tau),   s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,
    tau  = (t - t0) / T

It starts and ends at rest (zero velocity and acceleration), moves in a
straight line from the start x0 to the end point xf, and is completely fixed
by x0, xf, the onset time t0 and the duration T. It is a *kinematic* model:
it describes the shape of the motion and ignores muscles, mass and forces.

How the predictor uses it, at every frame (only past samples are used):

1. Detect the movement onset from the causal speed (same rule as segmentation).
2. Start point x0 = median position while the hand rested before the onset.
3. End point xf = the predetermined target (known from calibration, part of
   the experimental design). The "free target" ablation fits xf instead.
4. Fit t0 and T to the samples observed so far: a coarse grid search, then
   bounded nonlinear least squares (scipy).
5. If the fit is not trustworthy yet (too few samples, or T is still very
   uncertain), fall back to constant velocity and record why.
6. Predict along the fitted curve; once tau >= 1 the hand holds at xf.

Two ways of evaluating the fitted model are provided, and they must agree
(tested): the closed-form polynomial, and the proposal's Newton-Euler stepping
v[n+1] = v[n] + a[n] dt, x[n+1] = x[n] + v[n] dt using the minimum-jerk
acceleration profile. Minimum jerk supplies the *shape* of the motion; Euler
stepping rolls the state forward. The Euler-rolled version is the headline.

Extra variant (an additional row in the results, NOT the headline):
``minimum_jerk_replan`` follows Hoff & Arbib (1993): at every moment a new
minimum-jerk path is planned from the hand's *current measured* position and
velocity to the target, finishing at the fitted end time. A real path is
slightly curved, so the straight-line model can be a little off sideways even
"now"; re-planning starts from where the hand really is. Mathematically the
re-planned path is the fitted curve plus two correction terms that fade to
zero at the end of the movement:

    x_replan(t+h) = x_mj(t+h) + dx * (1 - s(h/R)) + dv * R * phi_v(h/R)

where dx, dv are (measured - model) position and velocity now, R is the
remaining movement time, and phi_v(u) = u - 6u^3 + 8u^4 - 3u^5 is the
quintic that starts with unit slope and ends at rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

from ..kinematics import find_onset
from ..predictor import Prediction, Predictor
from ..trial import History
from .baselines import constant_velocity, euler_roll, horizon_steps

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# The minimum-jerk profile and its derivatives (in normalised time tau)
# -----------------------------------------------------------------------------
def mj_s(tau: np.ndarray) -> np.ndarray:
    """Normalised position s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 (0 before, 1 after the movement)."""
    tau = np.clip(tau, 0.0, 1.0)
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


def mj_ds(tau: np.ndarray) -> np.ndarray:
    """ds/dtau = 30 tau^2 - 60 tau^3 + 30 tau^4 (zero outside the movement)."""
    tau = np.asarray(tau, dtype=float)
    inside = (tau > 0.0) & (tau < 1.0)
    tc = np.clip(tau, 0.0, 1.0)
    return np.where(inside, 30.0 * tc**2 * (1.0 - tc) ** 2, 0.0)


def mj_dds(tau: np.ndarray) -> np.ndarray:
    """d^2s/dtau^2 = 60 tau - 180 tau^2 + 120 tau^3 (zero outside the movement)."""
    tau = np.asarray(tau, dtype=float)
    inside = (tau > 0.0) & (tau < 1.0)
    tc = np.clip(tau, 0.0, 1.0)
    return np.where(inside, 60.0 * tc * (1.0 - tc) * (1.0 - 2.0 * tc), 0.0)


@dataclass(frozen=True)
class MinJerkFit:
    """Result of fitting the minimum-jerk model to the history so far."""

    ok: bool
    reason: str | None = None  # why the fit is unusable (when ok is False)
    t0: float = np.nan
    duration: float = np.nan
    x0: np.ndarray | None = None
    xf: np.ndarray | None = None
    rel_duration_std: float = np.nan
    progress_now: float = np.nan
    n_samples: int = 0
    onset_index: int | None = None

    def tau(self, t: np.ndarray) -> np.ndarray:
        return (np.asarray(t, dtype=float) - self.t0) / self.duration

    def position(self, t: np.ndarray) -> np.ndarray:
        s = mj_s(self.tau(t))
        return self.x0 + np.multiply.outer(s, self.xf - self.x0)

    def velocity(self, t: np.ndarray) -> np.ndarray:
        ds = mj_ds(self.tau(t))
        return np.multiply.outer(ds / self.duration, self.xf - self.x0)

    def acceleration(self, t: np.ndarray) -> np.ndarray:
        dds = mj_dds(self.tau(t))
        return np.multiply.outer(dds / self.duration**2, self.xf - self.x0)


def phi_v(u: np.ndarray) -> np.ndarray:
    """Quintic with phi(0)=0, phi'(0)=1, phi''(0)=0 and phi=phi'=phi''=0 at u=1."""
    u = np.clip(u, 0.0, 1.0)
    return u - 6.0 * u**3 + 8.0 * u**4 - 3.0 * u**5


def phi_v_dd(u: np.ndarray) -> np.ndarray:
    """Second derivative of ``phi_v`` (zero outside [0, 1])."""
    u = np.asarray(u, dtype=float)
    inside = (u > 0.0) & (u < 1.0)
    uc = np.clip(u, 0.0, 1.0)
    return np.where(inside, -36.0 * uc + 96.0 * uc**2 - 60.0 * uc**3, 0.0)


@dataclass(frozen=True)
class ReplanTerms:
    """Correction from the fitted curve to the hand's measured state (Hoff & Arbib re-planning)."""

    dx: np.ndarray  # measured - model position now
    dv: np.ndarray  # measured - model velocity now
    remaining: float  # time left until the fitted end of the movement (at least min_remaining)


def replan_terms(fit: MinJerkFit, t_now: float, x_meas: np.ndarray, v_meas: np.ndarray, min_remaining: float) -> ReplanTerms:
    now = np.array([t_now])
    dx = np.asarray(x_meas, float) - fit.position(now)[0]
    dv = np.asarray(v_meas, float) - fit.velocity(now)[0]
    remaining = max(fit.t0 + fit.duration - t_now, min_remaining)
    return ReplanTerms(dx, dv, remaining)


def predict_replan_closed(fit: MinJerkFit, terms: ReplanTerms, t_now: float, horizons_s: np.ndarray) -> np.ndarray:
    """Closed-form re-planned path (used to check the Euler-rolled version)."""
    horizons_s = np.asarray(horizons_s, dtype=float)
    u = horizons_s / terms.remaining
    return (
        fit.position(t_now + horizons_s)
        + np.outer(1.0 - mj_s(u), terms.dx)
        + np.outer(terms.remaining * phi_v(u), terms.dv)
    )


def predict_replan_euler(fit: MinJerkFit, terms: ReplanTerms, t_now: float, horizons_s: np.ndarray, dt: float) -> np.ndarray:
    """Re-planned path rolled forward with Newton-Euler steps from the measured state."""
    horizons_s = np.asarray(horizons_s, dtype=float)
    steps = horizon_steps(horizons_s, dt)
    r = terms.remaining

    def accel_at(times: np.ndarray) -> np.ndarray:
        u = times / r
        return (
            fit.acceleration(t_now + times)
            + np.outer(-mj_dds(u) / r**2, terms.dx)
            + np.outer(phi_v_dd(u) / r, terms.dv)
        )

    x_start = fit.position(np.array([t_now]))[0] + terms.dx
    v_start = fit.velocity(np.array([t_now]))[0] + terms.dv
    path = euler_roll(x_start, v_start, accel_at, dt, int(steps.max()))
    out = path[steps]
    out[horizons_s >= r] = fit.xf  # re-planned movement finished: rest at the target
    return out


def predict_closed_form(fit: MinJerkFit, t_now: float, horizons_s: np.ndarray) -> np.ndarray:
    """Evaluate the fitted polynomial at now + horizon (holds at xf once tau >= 1)."""
    return fit.position(t_now + np.asarray(horizons_s))


def predict_euler_rolled(fit: MinJerkFit, t_now: float, horizons_s: np.ndarray, dt: float) -> np.ndarray:
    """Roll the fitted state forward with Newton-Euler steps using the minimum-jerk acceleration."""
    horizons_s = np.asarray(horizons_s, dtype=float)
    steps = horizon_steps(horizons_s, dt)
    x_now = fit.position(np.array([t_now]))[0]
    v_now = fit.velocity(np.array([t_now]))[0]
    path = euler_roll(x_now, v_now, lambda times: fit.acceleration(t_now + times), dt, int(steps.max()))
    out = path[steps]
    # The model says the movement is over once tau >= 1: hold at the end point.
    finished = fit.tau(t_now + horizons_s) >= 1.0
    out[finished] = fit.xf
    return out


# -----------------------------------------------------------------------------
# Online fitting
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MinJerkSettings:
    """Fitting settings (see ``physics.minimum_jerk`` in the config for explanations)."""

    onset_speed: float
    onset_min_frames: int
    min_samples_after_onset: int
    baseline_window_s: tuple[float, float]
    t0_before_onset_s: float
    t0_after_onset_s: float
    duration_bounds_s: tuple[float, float]
    grid_t0_points: int
    grid_duration_points: int
    max_rel_duration_std: float
    min_progress: float
    free_target_min_progress: float
    loss: str

    @classmethod
    def from_config(cls, cfg: Mapping) -> "MinJerkSettings":
        mj = cfg["physics"]["minimum_jerk"]
        seg = cfg["kinematics"]["segmentation"]
        b_start, b_end = mj["baseline_window_ms"]
        return cls(
            onset_speed=seg["onset_speed_cm_s"],
            onset_min_frames=seg["onset_min_frames"],
            min_samples_after_onset=mj["min_samples_after_onset"],
            baseline_window_s=(b_start / 1000.0, b_end / 1000.0),
            t0_before_onset_s=mj["t0_search_before_onset_ms"] / 1000.0,
            t0_after_onset_s=mj["t0_search_after_onset_ms"] / 1000.0,
            duration_bounds_s=tuple(mj["duration_bounds_s"]),
            grid_t0_points=mj["grid_t0_points"],
            grid_duration_points=mj["grid_duration_points"],
            max_rel_duration_std=mj["max_relative_duration_std"],
            min_progress=mj["min_progress"],
            free_target_min_progress=mj["free_target_min_progress"],
            loss=mj["loss"],
        )


class MinimumJerkFitter:
    """Fits t0 and T (and, for the free-target ablation, also xf) to a History.

    Fitting is the expensive part, and the closed-form and Euler-rolled
    predictors need the same fit, so the last few fits are remembered for the
    exact same History object.
    """

    def __init__(self, settings: MinJerkSettings, free_target: bool = False) -> None:
        self.s = settings
        self.free_target = free_target
        self._cache: dict[int, tuple[History, MinJerkFit]] = {}

    def fit(self, history: History) -> MinJerkFit:
        cached = self._cache.get(id(history))
        if cached is not None and cached[0] is history:
            return cached[1]
        result = self._fit(history)
        if len(self._cache) > 16:
            self._cache.clear()
        self._cache[id(history)] = (history, result)
        return result

    # -- the actual fit ------------------------------------------------------
    def _fit(self, h: History) -> MinJerkFit:
        s = self.s
        if not self.free_target and h.context.target_xy is None:
            return MinJerkFit(False, "no_target")
        onset = find_onset(h.speed, s.onset_speed, s.onset_min_frames)
        if onset is None:
            return MinJerkFit(False, "no_onset_yet")
        measured = h.measured & np.all(np.isfinite(h.xy), axis=1)
        if measured[onset:].sum() < s.min_samples_after_onset:
            return MinJerkFit(False, "too_few_samples", onset_index=onset)

        t_on = h.t[onset]
        x0 = self._start_point(h, measured, t_on)
        if x0 is None:
            return MinJerkFit(False, "no_baseline", onset_index=onset)

        use = measured & (h.t >= t_on - s.t0_before_onset_s - 0.05)
        t, xy = h.t[use], h.xy[use]
        t0_lo, t0_hi = t_on - s.t0_before_onset_s, min(t_on + s.t0_after_onset_s, h.now)
        d_lo, d_hi = s.duration_bounds_s

        if self.free_target:
            result = self._fit_free_target(t, xy, x0, (t0_lo, t0_hi), (d_lo, d_hi))
        else:
            result = self._fit_known_target(t, xy, x0, np.asarray(h.context.target_xy, float), (t0_lo, t0_hi), (d_lo, d_hi))
        if result is None:
            return MinJerkFit(False, "degenerate", onset_index=onset)
        t0, duration, xf, rel_std = result

        progress = float(mj_s(np.array([(h.now - t0) / duration]))[0])
        fit = MinJerkFit(True, None, t0, duration, x0, xf, rel_std, progress, int(len(t)), onset)
        min_progress = s.free_target_min_progress if self.free_target else s.min_progress
        if not np.isfinite(rel_std) or rel_std > s.max_rel_duration_std:
            return _replace_reason(fit, "ill_conditioned")
        if progress < min_progress:
            return _replace_reason(fit, "too_early")
        return fit

    def _start_point(self, h: History, measured: np.ndarray, t_on: float) -> np.ndarray | None:
        """Median position while the hand was resting before the onset."""
        b_start, b_end = self.s.baseline_window_s
        window = measured & (h.t >= t_on - b_start) & (h.t <= t_on - b_end)
        if window.sum() < 3:
            window = measured & (h.t < t_on)
        if window.sum() < 3:
            return None
        return np.median(h.xy[window], axis=0)

    def _grid(self, t0_bounds, d_bounds) -> tuple[np.ndarray, np.ndarray]:
        t0_grid = np.linspace(t0_bounds[0], t0_bounds[1], self.s.grid_t0_points)
        d_grid = np.geomspace(d_bounds[0], d_bounds[1], self.s.grid_duration_points)
        return t0_grid, d_grid

    @staticmethod
    def _grid_samples(n: int, max_points: int = 40) -> np.ndarray:
        """Evenly spaced subset of samples for the coarse grid search.

        The grid only has to land near the best (t0, T); the least-squares
        refinement afterwards uses every sample, so this saves time without
        changing the answer.
        """
        if n <= max_points:
            return np.arange(n)
        return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))

    def _fit_known_target(self, t, xy, x0, xf, t0_bounds, d_bounds):
        delta = xf - x0
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            return None
        # Progress along the start -> target line (0 at the start, 1 at the
        # target). Sideways deviations do not depend on t0 or T, so fitting
        # this 1-D progress is the same as fitting the 2-D positions.
        progress = (xy - x0) @ delta / length**2

        t0_grid, d_grid = self._grid(t0_bounds, d_bounds)
        g = self._grid_samples(len(t))
        tau = (t[g][None, None, :] - t0_grid[:, None, None]) / d_grid[None, :, None]
        sse = np.sum((progress[g][None, None, :] - mj_s(tau)) ** 2, axis=2)
        i, j = np.unravel_index(np.argmin(sse), sse.shape)

        def residuals(p):
            return length * (progress - mj_s((t - p[0]) / p[1]))

        def jacobian(p):
            tau_ = (t - p[0]) / p[1]
            ds = mj_ds(tau_)
            return np.column_stack([length * ds / p[1], length * ds * tau_ / p[1]])

        res = least_squares(
            residuals,
            x0=[t0_grid[i], d_grid[j]],
            jac=jacobian,
            bounds=([t0_bounds[0], d_bounds[0]], [max(t0_bounds[1], t0_bounds[0] + 1e-6), d_bounds[1]]),
            loss=self.s.loss,
            x_scale=[0.05, 0.3],
            method="trf",
        )
        t0, duration = float(res.x[0]), float(res.x[1])
        rel_std = _relative_std(jacobian(res.x), residuals(res.x), param_index=1, value=duration)
        return t0, duration, xf.copy(), rel_std

    def _fit_free_target(self, t, xy, x0, t0_bounds, d_bounds):
        rel = xy - x0
        t0_grid, d_grid = self._grid(t0_bounds, d_bounds)
        g = self._grid_samples(len(t))
        tau = (t[g][None, None, :] - t0_grid[:, None, None]) / d_grid[None, :, None]
        s_vals = mj_s(tau)
        # For fixed t0 and T the best end point has a closed-form answer
        # (linear least squares), so the grid only searches over t0 and T.
        s2 = np.sum(s_vals**2, axis=2)
        sx = np.einsum("abn,nk->abk", s_vals, rel[g])
        with np.errstate(invalid="ignore", divide="ignore"):
            sse = np.sum(rel[g] ** 2) - np.sum(sx**2, axis=2) / s2
        sse = np.where(s2 > 1e-9, sse, np.inf)
        if not np.isfinite(sse).any():
            return None
        i, j = np.unravel_index(np.argmin(sse), sse.shape)
        delta0 = sx[i, j] / s2[i, j]

        def residuals(p):
            s_ = mj_s((t - p[0]) / p[1])
            return (rel - np.outer(s_, p[2:4])).ravel()

        def jacobian(p):
            tau_ = (t - p[0]) / p[1]
            s_, ds = mj_s(tau_), mj_ds(tau_)
            d = p[2:4]
            jac = np.zeros((len(t), 2, 4))
            jac[:, :, 0] = np.outer(ds / p[1], d)
            jac[:, :, 1] = np.outer(ds * tau_ / p[1], d)
            jac[:, 0, 2] = -s_
            jac[:, 1, 3] = -s_
            return jac.reshape(2 * len(t), 4)

        lo = [t0_bounds[0], d_bounds[0], -np.inf, -np.inf]
        hi = [max(t0_bounds[1], t0_bounds[0] + 1e-6), d_bounds[1], np.inf, np.inf]
        res = least_squares(
            residuals,
            x0=[t0_grid[i], d_grid[j], delta0[0], delta0[1]],
            jac=jacobian,
            bounds=(lo, hi),
            loss=self.s.loss,
            x_scale=[0.05, 0.3, 5.0, 5.0],
            method="trf",
        )
        t0, duration = float(res.x[0]), float(res.x[1])
        xf = x0 + res.x[2:4]
        rel_std = _relative_std(jacobian(res.x), residuals(res.x), param_index=1, value=duration)
        return t0, duration, xf, rel_std


def _relative_std(jac: np.ndarray, resid: np.ndarray, param_index: int, value: float) -> float:
    """Standard error of one fitted parameter divided by its value (from the Jacobian).

    The covariance of least-squares estimates is about sigma^2 (J^T J)^-1, with
    sigma^2 estimated from the residuals. A large relative error on T means the
    data seen so far cannot tell a short movement from a long one yet.
    """
    n, k = jac.shape
    if n <= k:
        return np.inf
    jtj = jac.T @ jac
    if not np.all(np.isfinite(jtj)) or np.linalg.cond(jtj) > 1e12:
        return np.inf
    sigma2 = float(np.sum(resid**2)) / (n - k)
    cov = sigma2 * np.linalg.inv(jtj)
    var = cov[param_index, param_index]
    return float(np.sqrt(max(var, 0.0)) / abs(value)) if value else np.inf


def _replace_reason(fit: MinJerkFit, reason: str) -> MinJerkFit:
    return MinJerkFit(
        False, reason, fit.t0, fit.duration, fit.x0, fit.xf, fit.rel_duration_std,
        fit.progress_now, fit.n_samples, fit.onset_index,
    )


# -----------------------------------------------------------------------------
# Predictors
# -----------------------------------------------------------------------------
class MinimumJerkPredictor(Predictor):
    """Predicts along the fitted minimum-jerk curve; falls back to constant velocity if needed.

    ``mode``: "closed" (polynomial), "euler" (Newton-Euler stepping, the
    headline) or "replan" (Hoff & Arbib re-planning from the measured state,
    also rolled forward with Euler steps).
    """

    def __init__(
        self,
        name: str,
        fitter: MinimumJerkFitter,
        mode: str = "euler",
        euler_dt_s: float = 1e-4,
        replan_min_remaining_s: float = 0.1,
    ) -> None:
        if mode not in ("closed", "euler", "replan"):
            raise ValueError("mode must be 'closed', 'euler' or 'replan'")
        self.name = name
        self.fitter = fitter
        self.mode = mode
        self.euler_dt = float(euler_dt_s)
        self.replan_min_remaining = float(replan_min_remaining_s)

    def predict(self, history: History, horizons_s: np.ndarray) -> Prediction:
        fit = self.fitter.fit(history)
        if not fit.ok:
            return Prediction(constant_velocity(history, horizons_s), fallback=fit.reason)
        if self.mode == "closed":
            positions = predict_closed_form(fit, history.now, horizons_s)
        elif self.mode == "euler":
            positions = predict_euler_rolled(fit, history.now, horizons_s, self.euler_dt)
        else:
            terms = replan_terms(fit, history.now, history.filt_xy[-1], history.filt_v[-1], self.replan_min_remaining)
            positions = predict_replan_euler(fit, terms, history.now, horizons_s, self.euler_dt)
        info = {"t0": fit.t0, "duration": fit.duration, "progress": fit.progress_now}
        return Prediction(positions, info=info)

    def describe(self) -> dict:
        return {
            **super().describe(),
            "mode": self.mode,
            "free_target": self.fitter.free_target,
            "euler_dt_s": self.euler_dt if self.mode != "closed" else None,
            "replan_min_remaining_s": self.replan_min_remaining if self.mode == "replan" else None,
        }
