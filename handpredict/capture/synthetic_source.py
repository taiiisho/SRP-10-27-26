"""A pretend camera that "films" synthetic reaching movements.

For tests, demos and ``handpredict selftest`` only. Each frame carries the true
hand position in ``meta["true_xy_px"]``; the synthetic landmark detector adds
realistic noise and missed detections to it. Every session recorded from this
source is labelled ``synthetic`` in its folder name, file names and contents.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from ..synthetic import ReachSpec, reach_position, sample_reach_spec
from .base import CapturedFrame, FrameSource, monotonic_ns


class SyntheticSource(FrameSource):
    """Generates frame timestamps and true hand positions for synthetic reaches."""

    name = "synthetic"
    is_synthetic = True

    def __init__(self, cfg: Mapping, seed: int | None = None, render: bool = False) -> None:
        cam = cfg["camera"]
        super().__init__(cam["width"], cam["height"], cam["fps"])
        self.syn = cfg["synthetic"]
        self.seed = cfg["study"]["seed"] if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.render = render
        self._t_s = 0.0
        self._index = 0
        self._spec: ReachSpec | None = None
        self._rest_px = np.asarray(self.syn["start_px"], dtype=float)
        self._open = False

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def current_spec(self) -> ReachSpec | None:
        return self._spec

    def on_trial_start(self, trial_info: dict[str, Any]) -> None:
        """Start a new reach: the hand rests in the start region, then moves after 'GO'."""
        go_time = self._t_s + float(trial_info.get("pre_roll_s", 0.5))
        self._spec = sample_reach_spec(
            self.rng,
            self.syn,
            target_id=trial_info["target_id"],
            speed_condition=trial_info["speed_condition"],
            go_time_s=go_time,
        )

    def _true_position(self, t_s: float) -> np.ndarray:
        if self._spec is None:
            return self._rest_px.copy()
        return reach_position(self._spec, np.array([t_s]))[0]

    def read(self, timeout_s: float = 1.0) -> CapturedFrame | None:
        if not self._open:
            raise RuntimeError("SyntheticSource.read() called before open()")
        interval = 1.0 / self.nominal_fps
        # Simulate occasional dropped frames (the camera skips a frame).
        while self.rng.random() < self.syn["dropped_frame_probability"]:
            self._t_s += interval
        jitter = self.rng.normal(0.0, self.syn["timing_jitter_ms"] / 1000.0)
        t_capture = max(self._t_s + jitter, self._t_s - 0.4 * interval)
        if self.syn.get("realtime", False):
            time.sleep(interval)
        xy = self._true_position(t_capture)
        frame = CapturedFrame(
            index=self._index,
            t_ns=int(round(t_capture * 1e9)),
            host_ns=monotonic_ns(),
            image=self._render(xy) if self.render else None,
            timestamp_source="synthetic",
            meta={"true_xy_px": xy, "synthetic": True},
        )
        self._index += 1
        self._t_s += interval
        return frame

    def _render(self, xy: np.ndarray) -> np.ndarray:
        """A plain grey image with a bright disc at the hand position (for the demo/preview)."""
        img = np.full((self.height, self.width, 3), 60, dtype=np.uint8)
        yy, xx = np.ogrid[: self.height, : self.width]
        mask = (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 <= 30**2
        img[mask] = (200, 180, 160)
        return img

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({"seed": self.seed, "timestamp_source": "synthetic"})
        return info
