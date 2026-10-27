"""Synthetic reaching movements, for tests, demos and ``handpredict selftest`` ONLY.

Synthetic data is never evidence for or against the hypothesis. It exists so
that every part of the software can be tested on a laptop without a camera,
and so that we know the exact true path the hand took.

To avoid giving the minimum-jerk model an unfair advantage, the synthetic
reaches deliberately differ from a perfect minimum-jerk movement in the ways
real reaches do:

* the path bows sideways (``curvature``), real reaches are slightly curved;
* the speed peak comes a little early (``asymmetry``);
* the hand stops near, not exactly on, the target (endpoint scatter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .kinematics import fill_short_gaps
from .trial import TrialContext, TrialData

SYNTHETIC_LABEL = "synthetic"


@dataclass(frozen=True)
class ReachSpec:
    """Parameters of one synthetic reach, in pixel coordinates and seconds."""

    start_px: np.ndarray
    target_px: np.ndarray  # nominal target the participant aims for
    endpoint_px: np.ndarray  # where the hand actually stops (target + scatter)
    onset_s: float  # movement onset, measured from the start of the recording
    duration_s: float
    curvature: float  # sideways bow as a fraction of the reach distance (signed)
    asymmetry: float  # 0 = symmetric minimum-jerk speed profile, >0 = earlier peak

    def as_dict(self) -> dict:
        return {
            "start_px": self.start_px.tolist(),
            "target_px": self.target_px.tolist(),
            "endpoint_px": self.endpoint_px.tolist(),
            "onset_s": self.onset_s,
            "duration_s": self.duration_s,
            "curvature": self.curvature,
            "asymmetry": self.asymmetry,
        }


def minimum_jerk_profile(tau: np.ndarray) -> np.ndarray:
    """Flash & Hogan normalised position s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 (clipped to [0, 1])."""
    tau = np.clip(tau, 0.0, 1.0)
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


def reach_position(spec: ReachSpec, t_s: np.ndarray) -> np.ndarray:
    """True hand position (pixels) at the times ``t_s`` for a synthetic reach."""
    t_s = np.atleast_1d(np.asarray(t_s, dtype=float))
    tau = np.clip((t_s - spec.onset_s) / spec.duration_s, 0.0, 1.0)
    # A smooth time warp w(tau) with w(0)=0, w(1)=1 and w' > 0 moves the speed
    # peak earlier when asymmetry > 0 (requires |asymmetry| < 1).
    warped = tau + spec.asymmetry * np.sin(np.pi * tau) / np.pi
    s = minimum_jerk_profile(warped)
    delta = spec.endpoint_px - spec.start_px
    distance = float(np.linalg.norm(delta))
    direction = delta / distance if distance > 0 else np.array([1.0, 0.0])
    normal = np.array([-direction[1], direction[0]])
    bow = spec.curvature * distance * np.sin(np.pi * s)
    return spec.start_px + s[:, None] * delta + bow[:, None] * normal


def sample_reach_spec(
    rng: np.random.Generator,
    syn_cfg: Mapping,
    target_id: str,
    speed_condition: str,
    go_time_s: float,
) -> ReachSpec:
    """Draw a random reach towards ``target_id`` at ``speed_condition`` speed."""
    start = np.asarray(syn_cfg["start_px"], dtype=float)
    start = start + rng.normal(0.0, syn_cfg["start_jitter_px"], size=2)
    target = np.asarray(syn_cfg["targets_px"][target_id], dtype=float)
    scatter_px = syn_cfg["endpoint_scatter_cm"] / syn_cfg["cm_per_px"]
    endpoint = target + rng.normal(0.0, scatter_px, size=2)
    lo, hi = syn_cfg["durations_s"][speed_condition]
    duration = float(rng.uniform(lo, hi))
    rt_lo, rt_hi = syn_cfg["reaction_time_s"]
    onset = go_time_s + float(rng.uniform(rt_lo, rt_hi))
    c_lo, c_hi = syn_cfg["curvature_fraction"]
    curvature = float(rng.uniform(c_lo, c_hi)) * (1.0 if rng.random() < 0.7 else -1.0)
    a_lo, a_hi = syn_cfg["asymmetry"]
    asymmetry = float(rng.uniform(a_lo, a_hi))
    return ReachSpec(start, target, endpoint, onset, duration, curvature, asymmetry)


def frame_times(
    rng: np.random.Generator,
    fps: float,
    total_s: float,
    jitter_ms: float = 0.0,
    drop_probability: float = 0.0,
) -> np.ndarray:
    """Capture times of a camera running at ``fps`` with small timing jitter and dropped frames."""
    n = int(np.floor(total_s * fps)) + 1
    t = np.arange(n) / fps
    if jitter_ms > 0:
        t = t + rng.normal(0.0, jitter_ms / 1000.0, size=n)
        t[0] = 0.0
        t = np.maximum.accumulate(t)
        t[1:] = np.maximum(t[1:], t[:-1] + 1e-4)  # keep strictly increasing
    if drop_probability > 0:
        keep = rng.random(n) >= drop_probability
        keep[0] = True
        t = t[keep]
    return t


def make_trial_data(
    spec: ReachSpec,
    cfg: Mapping,
    rng: np.random.Generator,
    trial_id: str,
    speed_condition: str,
    target_id: str,
    session_id: str = "synthetic_session",
    noise_px: float = 0.0,
    miss_probability: float = 0.0,
    total_s: float | None = None,
    timing: bool = True,
) -> TrialData:
    """A complete synthetic trial in centimetres, as if recorded and detected.

    ``noise_px`` / ``miss_probability`` imitate the detector; ``timing``
    switches camera timing jitter and dropped frames on or off.
    """
    syn = cfg["synthetic"]
    rec = cfg["recording"]
    fps = float(cfg["camera"]["fps"])
    total = total_s if total_s is not None else rec["pre_roll_s"] + rec["duration_s"]
    jitter = syn["timing_jitter_ms"] if timing else 0.0
    drop = syn["dropped_frame_probability"] if timing else 0.0
    t = frame_times(rng, fps, total, jitter, drop)
    xy_px = reach_position(spec, t)
    if noise_px > 0:
        xy_px = xy_px + rng.normal(0.0, noise_px, xy_px.shape)
    detected = rng.random(len(t)) >= miss_probability
    filled, interpolated = fill_short_gaps(t, xy_px, detected, rec["max_gap_frames"])
    cm_per_px = float(syn["cm_per_px"])
    context = TrialContext(
        trial_id=trial_id,
        session_id=session_id,
        participant_id=syn["participant_id"],
        speed_condition=speed_condition,
        target_id=target_id,
        start_xy=np.asarray(syn["start_px"], float) * cm_per_px,
        target_xy=spec.target_px * cm_per_px,
        units="cm",
        fps=fps,
        data_origin=SYNTHETIC_LABEL,
        cm_per_px=cm_per_px,
        extra={"reach_spec": spec.as_dict()},
    )
    return TrialData(context, t, filled * cm_per_px, detected, interpolated)


def render_hand_video(
    hand_image_bgr: np.ndarray,
    out_path: str,
    path_px: np.ndarray,
    fps: float,
    size: tuple[int, int] = (1280, 720),
    hand_height_px: int = 240,
    codec: str = "mp4v",
) -> None:
    """Write a SEMI-SYNTHETIC video: a real hand photo moved along ``path_px``.

    Used to test the real detector (MediaPipe) on a movement whose true path
    is known exactly. The photo's centre follows ``path_px`` (one row per frame).
    """
    import cv2

    scale = hand_height_px / hand_image_bgr.shape[0]
    hand = cv2.resize(hand_image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hh, hw = hand.shape[:2]
    width, height = size
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {out_path}")
    try:
        for cx, cy in np.asarray(path_px, dtype=float):
            canvas = np.full((height, width, 3), 90, dtype=np.uint8)
            x0, y0 = int(round(cx - hw / 2)), int(round(cy - hh / 2))
            xs, ys = max(x0, 0), max(y0, 0)
            xe, ye = min(x0 + hw, width), min(y0 + hh, height)
            if xe > xs and ye > ys:
                canvas[ys:ye, xs:xe] = hand[ys - y0 : ye - y0, xs - x0 : xe - x0]
            writer.write(canvas)
    finally:
        writer.release()


def trial_plan(speeds, targets, repetitions: int, rng: np.random.Generator | None) -> list[tuple[str, str]]:
    """Every (speed, target) combination ``repetitions`` times; shuffled if an rng is given."""
    plan = [(s, tgt) for _ in range(repetitions) for s in speeds for tgt in targets]
    if rng is not None:
        order = rng.permutation(len(plan))
        plan = [plan[i] for i in order]
    return plan
