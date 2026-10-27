"""In-memory trial data and the "history" handed to every predictor.

A predictor never receives a whole trial. At frame j it receives a
``History``: copies of the samples 0..j only. Because the future samples are
not inside the object at all, a predictor cannot accidentally look ahead.
Physics models and ML models receive exactly the same History objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .kinematics import CausalFilter, KinematicState, Segmentation, segment_from_config


@dataclass(frozen=True)
class TrialContext:
    """Facts about a trial that are known before the movement starts."""

    trial_id: str
    session_id: str
    participant_id: str
    speed_condition: str
    target_id: str
    start_xy: np.ndarray | None  # centre of the calibrated start region (analysis units)
    target_xy: np.ndarray | None  # calibrated target location (analysis units)
    units: str  # "cm" (calibrated) or "px"
    fps: float  # nominal camera frame rate
    data_origin: str = "real"  # "real" or "synthetic"
    cm_per_px: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class TrialData:
    """One trial as measured: times, positions and which frames had a real detection."""

    context: TrialContext
    t: np.ndarray  # (n,) seconds since the start of the recording
    xy: np.ndarray  # (n, 2) positions, short gaps interpolated, NaN in long gaps
    measured: np.ndarray  # (n,) True where the detector really saw the hand
    interpolated: np.ndarray | None = None  # (n,) True where a short gap was filled

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.xy = np.asarray(self.xy, dtype=float)
        self.measured = np.asarray(self.measured, dtype=bool)
        if self.interpolated is None:
            self.interpolated = np.zeros(len(self.t), dtype=bool)
        self.interpolated = np.asarray(self.interpolated, dtype=bool)
        n = len(self.t)
        if self.xy.shape != (n, 2) or self.measured.shape != (n,) or self.interpolated.shape != (n,):
            raise ValueError("TrialData arrays have inconsistent shapes")
        if np.any(self.measured & self.interpolated):
            raise ValueError("a frame cannot be both measured and interpolated")
        if n > 1 and np.any(np.diff(self.t) <= 0):
            raise ValueError(f"trial {self.context.trial_id}: timestamps must be strictly increasing")

    @property
    def n(self) -> int:
        return len(self.t)


@dataclass
class ProcessedTrial:
    """A trial plus its causal kinematics and movement segmentation."""

    data: TrialData
    kin: KinematicState
    segmentation: Segmentation

    @property
    def context(self) -> TrialContext:
        return self.data.context

    @property
    def speed(self) -> np.ndarray:
        return self.kin.speed


def process_trial(data: TrialData, causal_filter: CausalFilter, cfg: Mapping) -> ProcessedTrial:
    """Run the causal filter over a trial and segment the movement.

    The filter only sees real measurements (``data.measured``); interpolated
    frames are treated as missing so that no value built from a later frame
    can enter the filter early.
    """
    kin = causal_filter.run(data.t, data.xy, data.measured)
    seg = segment_from_config(kin.speed, cfg)
    return ProcessedTrial(data=data, kin=kin, segmentation=seg)


@dataclass(frozen=True)
class History:
    """Everything a predictor may know at frame ``index``: samples 0..index (copies)."""

    index: int
    t: np.ndarray
    xy: np.ndarray
    measured: np.ndarray
    filt_xy: np.ndarray
    filt_v: np.ndarray
    filt_a: np.ndarray
    context: TrialContext

    @property
    def now(self) -> float:
        return float(self.t[-1])

    @property
    def current_xy(self) -> np.ndarray:
        """The latest measured position (the anchor for 'stay where you are')."""
        return self.xy[-1]

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.filt_v, axis=1)


def make_history(trial: ProcessedTrial, index: int) -> History:
    """Cut a trial at frame ``index``; arrays are copied so no future data is reachable."""
    if not 0 <= index < trial.data.n:
        raise IndexError(f"frame {index} outside trial of length {trial.data.n}")
    sl = slice(0, index + 1)
    return History(
        index=index,
        t=trial.data.t[sl].copy(),
        xy=trial.data.xy[sl].copy(),
        measured=trial.data.measured[sl].copy(),
        filt_xy=trial.kin.position[sl].copy(),
        filt_v=trial.kin.velocity[sl].copy(),
        filt_a=trial.kin.acceleration[sl].copy(),
        context=trial.context,
    )
