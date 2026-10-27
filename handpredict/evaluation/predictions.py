"""The prediction table: every model, every frame, every horizon, the same ground truth.

Fairness rules enforced here:

* All models are evaluated on EXACTLY the same frames: a frame is used only
  if every model can predict there (a real detection at that frame, a full
  ML input window without long gaps, an initialised causal filter).
* All models see the same History (samples up to now) and are compared with
  the same ground truth: the measured position at t + h, interpolated from
  real detections only.
* Every (frame, horizon) row records which analysis windows it belongs to:
    movement_strict      onset <= t <= offset and t + h <= offset   (primary)
    movement_any_target  onset <= t <= offset
    whole_trial          any frame with ground truth
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..kinematics import ground_truth_at
from ..ml.inference import WindowPredictor
from ..ml.windows import WindowSpec, build_windows
from ..predictor import Predictor
from ..trial import ProcessedTrial, make_history

log = logging.getLogger(__name__)

WINDOWS = ("movement_strict", "movement_any_target", "whole_trial")


def eligible_frames(trial: ProcessedTrial, spec: WindowSpec) -> np.ndarray:
    """Frames at which every model can make a prediction."""
    ws = build_windows(trial.data, spec)  # window exists => real detection + full history
    frames = ws.frame_index
    filt_ok = np.all(np.isfinite(trial.kin.position[frames]), axis=1) & np.all(np.isfinite(trial.kin.velocity[frames]), axis=1)
    return frames[filt_ok]


def predict_trial(
    trial: ProcessedTrial,
    physics: Mapping[str, Predictor],
    ml: Mapping[str, WindowPredictor],
    horizons_ms: Sequence[float],
    spec: WindowSpec,
    split: str,
    max_gap_s: float,
) -> pd.DataFrame:
    """All predictions for one trial in wide format: one row per (frame, horizon)."""
    data, kin, seg = trial.data, trial.kin, trial.segmentation
    ctx = trial.context
    horizons_s = np.asarray(horizons_ms, dtype=float) / 1000.0
    frames = eligible_frames(trial, spec)
    if len(frames) == 0:
        return pd.DataFrame()
    n_f, n_h = len(frames), len(horizons_s)
    t_now = data.t[frames]
    truth, valid = ground_truth_at(
        data.t[data.measured], data.xy[data.measured], (t_now[:, None] + horizons_s[None, :]).ravel(), max_gap_s
    )
    truth = truth.reshape(n_f, n_h, 2)
    valid = valid.reshape(n_f, n_h)

    preds: dict[str, np.ndarray] = {}
    fallbacks: dict[str, np.ndarray] = {}
    for name, model in physics.items():
        out = np.empty((n_f, n_h, 2))
        fb = np.empty(n_f, dtype=object)
        for i, j in enumerate(frames):
            p = model.predict(make_history(trial, int(j)), horizons_s)
            out[i] = p.positions
            fb[i] = p.fallback or ""
        preds[name] = out
        fallbacks[name] = fb
    if ml:
        ws = build_windows(data, spec, frames=frames)
        if not np.array_equal(ws.frame_index, frames):
            raise RuntimeError("ML windows do not match the eligible frames")
        anchor = data.xy[frames][:, None, :]
        for name, model in ml.items():
            preds[name] = anchor + model.predict_windowset(ws)

    in_move = np.zeros(n_f, dtype=bool)
    strict = np.zeros((n_f, n_h), dtype=bool)
    if seg.found:
        t_on, t_off = data.t[seg.onset_index], data.t[seg.offset_index]
        in_move = (t_now >= t_on) & (t_now <= t_off)
        strict = in_move[:, None] & (t_now[:, None] + horizons_s[None, :] <= t_off + 1e-9)
    speed = kin.speed[frames]
    scale = ctx.cm_per_px if ctx.cm_per_px else np.nan

    rows = {
        "trial_id": np.repeat(ctx.trial_id, n_f * n_h),
        "session_id": np.repeat(ctx.session_id, n_f * n_h),
        "participant_id": np.repeat(ctx.participant_id, n_f * n_h),
        "speed_condition": np.repeat(ctx.speed_condition, n_f * n_h),
        "target_id": np.repeat(ctx.target_id, n_f * n_h),
        "split": np.repeat(split, n_f * n_h),
        "data_origin": np.repeat(ctx.data_origin, n_f * n_h),
        "frame_index": np.repeat(frames, n_h),
        "timestamp": np.repeat(t_now, n_h),
        "x": np.repeat(data.xy[frames, 0], n_h),
        "y": np.repeat(data.xy[frames, 1], n_h),
        "vx": np.repeat(kin.velocity[frames, 0], n_h),
        "vy": np.repeat(kin.velocity[frames, 1], n_h),
        "ax": np.repeat(kin.acceleration[frames, 0], n_h),
        "ay": np.repeat(kin.acceleration[frames, 1], n_h),
        "speed": np.repeat(speed, n_h),
        "horizon_ms": np.tile(np.asarray(horizons_ms, dtype=float), n_f),
        "actual_x": truth[..., 0].ravel(),
        "actual_y": truth[..., 1].ravel(),
        "gt_valid": valid.ravel(),
        "in_movement": np.repeat(in_move, n_h),
        "win_movement_strict": (strict & valid).ravel(),
        "win_movement_any_target": (in_move[:, None] & valid).ravel(),
        "win_whole_trial": valid.ravel(),
        "units": np.repeat(ctx.units, n_f * n_h),
        "cm_per_px": np.repeat(scale, n_f * n_h),
    }
    for name, p in preds.items():
        err = np.linalg.norm(p - truth, axis=2)
        err[~valid] = np.nan
        rows[f"{name}_pred_x"] = p[..., 0].ravel()
        rows[f"{name}_pred_y"] = p[..., 1].ravel()
        rows[f"{name}_error"] = err.ravel()
        if name in fallbacks:
            rows[f"{name}_fallback"] = np.repeat(fallbacks[name], n_h)
    return pd.DataFrame(rows)


def add_headline_columns(df: pd.DataFrame, physics_name: str, ml_name: str) -> pd.DataFrame:
    """The column names used in the proposal: physics_*, ml_* and errors in cm and pixels."""
    df = df.copy()
    for prefix, name in (("physics", physics_name), ("ml", ml_name)):
        df[f"{prefix}_model"] = name
        df[f"{prefix}_pred_x"] = df[f"{name}_pred_x"]
        df[f"{prefix}_pred_y"] = df[f"{name}_pred_y"]
        df[f"{prefix}_error"] = df[f"{name}_error"]
        df[f"{prefix}_error_px"] = df[f"{name}_error"] / df["cm_per_px"]
    df["physics_fallback"] = df.get(f"{physics_name}_fallback", "")
    return df


def build_prediction_table(
    trials: Sequence[tuple[ProcessedTrial, str]],
    physics: Mapping[str, Predictor],
    ml: Mapping[str, WindowPredictor],
    horizons_ms: Sequence[float],
    spec: WindowSpec,
    max_gap_s: float,
) -> pd.DataFrame:
    parts = []
    for k, (trial, split) in enumerate(trials):
        parts.append(predict_trial(trial, physics, ml, horizons_ms, spec, split, max_gap_s))
        if (k + 1) % 10 == 0:
            log.info("  predicted %d / %d trials", k + 1, len(trials))
    parts = [p for p in parts if len(p)]
    if not parts:
        raise RuntimeError("no frame could be evaluated (check that the trials have movement and history)")
    return pd.concat(parts, ignore_index=True)
