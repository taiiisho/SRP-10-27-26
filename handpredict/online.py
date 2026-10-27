"""Frame-by-frame prediction for the live demo and the benchmark.

``OnlinePredictor`` receives one detection per frame and keeps the same
buffers the offline analysis uses (positions, causal filter states), so the
live predictions are computed by exactly the same code as the results
(``tests/test_online.py`` checks they are identical):

* the causal filter is updated one sample at a time;
* a short gap of missed detections is filled by linear interpolation as soon
  as the next real detection arrives (just like ``fill_short_gaps``, which
  also needs that next detection);
* predictors receive a ``History`` of everything up to now.

For long live sessions the buffers are trimmed: once the hand has been
resting after a movement, older samples are dropped, so the next movement
starts a fresh "trial" for the minimum-jerk onset detection.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .kinematics import find_onset, make_filter
from .predictor import Prediction, Predictor
from .trial import History, TrialContext


class OnlinePredictor:
    def __init__(
        self,
        cfg: Mapping,
        context: TrialContext,
        predictors: Mapping[str, Predictor],
        trim: bool = True,
        max_history_s: float = 10.0,
        rest_reset_s: float = 0.6,
    ) -> None:
        self.cfg = cfg
        self.context = context
        self.predictors = dict(predictors)
        self.stepper = make_filter(cfg).new_state()
        self.max_gap = int(cfg["recording"]["max_gap_frames"])
        seg = cfg["kinematics"]["segmentation"]
        self.onset_speed, self.onset_frames = seg["onset_speed_cm_s"], seg["onset_min_frames"]
        self.offset_speed = seg["offset_speed_cm_s"]
        self.trim = trim
        self.max_history_s = max_history_s
        self.rest_reset_s = rest_reset_s
        self.t: list[float] = []
        self.xy: list[np.ndarray] = []
        self.measured: list[bool] = []
        self.fxy: list[np.ndarray] = []
        self.fv: list[np.ndarray] = []
        self.fa: list[np.ndarray] = []
        self._rest_since: float | None = None

    def __len__(self) -> int:
        return len(self.t)

    def update(self, t_s: float, xy: np.ndarray | None) -> None:
        """Add one frame (``xy`` = None if the hand was not detected)."""
        measured = xy is not None and bool(np.all(np.isfinite(xy)))
        z = np.asarray(xy, dtype=float) if measured else np.full(2, np.nan)
        if measured:
            self._fill_gap_before(float(t_s), z)
        p, v, a, _ = self.stepper.update(float(t_s), z, measured)
        self.t.append(float(t_s))
        self.xy.append(z.copy())
        self.measured.append(measured)
        self.fxy.append(p)
        self.fv.append(v)
        self.fa.append(a)
        if self.trim:
            self._trim()

    def _fill_gap_before(self, t_new: float, z_new: np.ndarray) -> None:
        """Interpolate a just-ended gap of <= max_gap missed frames (only past frames change)."""
        k = 0
        while k < len(self.measured) and not self.measured[-1 - k]:
            k += 1
        if k == 0 or k > self.max_gap or k == len(self.measured):
            return
        left = len(self.measured) - 1 - k
        t0, z0 = self.t[left], self.xy[left]
        for i in range(left + 1, len(self.measured)):
            w = (self.t[i] - t0) / (t_new - t0)
            self.xy[i] = z0 + w * (z_new - z0)

    def _trim(self) -> None:
        if self.t and self.t[-1] - self.t[0] > self.max_history_s:
            self._drop_before(self.t[-1] - self.max_history_s)
        speed = float(np.hypot(*self.fv[-1])) if np.all(np.isfinite(self.fv[-1])) else 0.0
        if speed < self.offset_speed:
            self._rest_since = self.t[-1] if self._rest_since is None else self._rest_since
        else:
            self._rest_since = None
        if self._rest_since is not None and self.t[-1] - self._rest_since >= self.rest_reset_s:
            moved = find_onset(np.linalg.norm(np.nan_to_num(np.asarray(self.fv)), axis=1), self.onset_speed, self.onset_frames)
            if moved is not None:  # a movement has finished: start a fresh baseline
                self._drop_before(self.t[-1] - self.rest_reset_s)

    def _drop_before(self, t_cut: float) -> None:
        keep = next((i for i, t in enumerate(self.t) if t >= t_cut), len(self.t))
        for buf in (self.t, self.xy, self.measured, self.fxy, self.fv, self.fa):
            del buf[:keep]

    def history(self) -> History:
        return History(
            index=len(self.t) - 1,
            t=np.asarray(self.t, dtype=float),
            xy=np.asarray(self.xy, dtype=float),
            measured=np.asarray(self.measured, dtype=bool),
            filt_xy=np.asarray(self.fxy, dtype=float),
            filt_v=np.asarray(self.fv, dtype=float),
            filt_a=np.asarray(self.fa, dtype=float),
            context=self.context,
        )

    def predict(self, horizons_s: np.ndarray, names: list[str] | None = None) -> dict[str, Prediction]:
        """Predictions of the selected models from the current history (None if not possible yet)."""
        if not self.t or not self.measured[-1] or not np.all(np.isfinite(self.fxy[-1])):
            return {}
        h = self.history()
        return {name: self.predictors[name].predict(h, horizons_s) for name in (names or list(self.predictors))}
