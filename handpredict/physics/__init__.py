"""Physics-based predictors.

``make_physics_predictors(cfg)`` builds every model listed in
``cfg["physics"]["models"]``. The headline model (``cfg["physics"]["headline"]``)
was fixed before any test data was looked at.
"""

from __future__ import annotations

from typing import Mapping

from ..predictor import Predictor
from .baselines import ConstantVelocityPredictor, EulerConstantAccelerationPredictor, ZeroMotionPredictor
from .minimum_jerk import MinimumJerkFitter, MinimumJerkPredictor, MinJerkSettings

PHYSICS_MODEL_NAMES = (
    "zero_motion",
    "constant_velocity",
    "euler_constant_acceleration",
    "minimum_jerk_closed",
    "minimum_jerk_euler",
    "minimum_jerk_replan",
    "minimum_jerk_free_target",
)


def make_physics_predictors(cfg: Mapping) -> dict[str, Predictor]:
    """Create the configured physics models (closed-form and Euler minimum jerk share one fitter)."""
    names = list(cfg["physics"]["models"])
    unknown = set(names) - set(PHYSICS_MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown physics models in config: {sorted(unknown)}")
    settings = MinJerkSettings.from_config(cfg)
    known_target_fitter = MinimumJerkFitter(settings, free_target=False)
    free_target_fitter = MinimumJerkFitter(settings, free_target=True)
    euler = cfg["physics"]["euler"]
    tau_ms = euler["acceleration_decay_tau_ms"]
    mj_cfg = cfg["physics"]["minimum_jerk"]
    mj_dt = mj_cfg["euler_dt_ms"] / 1000.0
    min_remaining = mj_cfg["replan_min_remaining_ms"] / 1000.0

    builders = {
        "zero_motion": lambda: ZeroMotionPredictor(),
        "constant_velocity": lambda: ConstantVelocityPredictor(),
        "euler_constant_acceleration": lambda: EulerConstantAccelerationPredictor(
            dt_s=euler["dt_ms"] / 1000.0, decay_tau_s=None if tau_ms is None else tau_ms / 1000.0
        ),
        "minimum_jerk_closed": lambda: MinimumJerkPredictor("minimum_jerk_closed", known_target_fitter, "closed"),
        "minimum_jerk_euler": lambda: MinimumJerkPredictor("minimum_jerk_euler", known_target_fitter, "euler", mj_dt),
        "minimum_jerk_replan": lambda: MinimumJerkPredictor(
            "minimum_jerk_replan", known_target_fitter, "replan", mj_dt, replan_min_remaining_s=min_remaining
        ),
        # The free-target ablation is rolled forward with Euler steps like the headline model.
        "minimum_jerk_free_target": lambda: MinimumJerkPredictor(
            "minimum_jerk_free_target", free_target_fitter, "euler", mj_dt
        ),
    }
    return {name: builders[name]() for name in names}


__all__ = [
    "ConstantVelocityPredictor",
    "EulerConstantAccelerationPredictor",
    "MinimumJerkFitter",
    "MinimumJerkPredictor",
    "MinJerkSettings",
    "PHYSICS_MODEL_NAMES",
    "ZeroMotionPredictor",
    "make_physics_predictors",
]
