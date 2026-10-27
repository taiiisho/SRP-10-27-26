"""Replay a video file frame by frame (deterministic, for re-running a detector).

When the recorder saves video (``--save-video``), every captured frame is in
the .mp4 and its original timestamp is in the trial CSV, so a replay uses the
real capture times, not the nominal frame rate. For other videos the
timestamps come from the file (``CAP_PROP_POS_MSEC``) or, failing that, from
frame index / fps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import CapturedFrame, FrameSource, SourceUnavailableError, monotonic_ns

log = logging.getLogger(__name__)


class VideoFileSource(FrameSource):
    """Frames from a video file, with optional exact timestamps (nanoseconds) per frame."""

    name = "video"

    def __init__(self, path: str | Path, timestamps_ns: Sequence[int] | None = None, fps: float | None = None) -> None:
        self.path = Path(path)
        self._timestamps = None if timestamps_ns is None else np.asarray(timestamps_ns, dtype=np.int64)
        self._requested_fps = fps
        super().__init__(0, 0, fps or 0.0)
        self._cap = None
        self._index = 0

    def open(self) -> None:
        import cv2

        if not self.path.exists():
            raise SourceUnavailableError(f"Video file {self.path} does not exist")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise SourceUnavailableError(f"OpenCV cannot read {self.path}")
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.nominal_fps = float(self._requested_fps or file_fps or 30.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._timestamps is not None and n_frames and abs(n_frames - len(self._timestamps)) > 0:
            log.warning("%s has %d frames but %d timestamps were given", self.path.name, n_frames, len(self._timestamps))

    def read(self, timeout_s: float = 1.0) -> CapturedFrame | None:
        import cv2

        if self._cap is None:
            raise RuntimeError("VideoFileSource.read() called before open()")
        ok, image = self._cap.read()
        if not ok:
            return None
        if self._timestamps is not None:
            if self._index >= len(self._timestamps):
                log.warning("More frames in %s than timestamps; stopping", self.path.name)
                return None
            t_ns = int(self._timestamps[self._index])
            source = "video (original capture timestamps)"
        else:
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            t_ns = int(round(pos_ms * 1e6)) if pos_ms > 0 or self._index == 0 else int(round(self._index * 1e9 / self.nominal_fps))
            source = "video (file timestamps)"
        frame = CapturedFrame(self._index, t_ns, monotonic_ns(), image, source, {})
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({"path": str(self.path), "exact_timestamps": self._timestamps is not None})
        return info
