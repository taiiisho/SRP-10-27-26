"""The common interface of every frame source (camera, video file, synthetic).

Timestamps matter more than anything else in this project: velocities,
accelerations and "where will the hand be in 50 ms" all depend on knowing when
each frame was taken. Rules:

* Use the camera's own sensor timestamp when it has one (Picamera2 does).
* Otherwise take ``time.monotonic_ns()`` the moment the frame arrives.
* Never use wall-clock time (``time.time()``): it can jump when the clock is
  synchronised over the network.

Every frame also carries ``host_ns`` (monotonic time when our code received
it), so the delay between capture and detection can be measured separately.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class SourceUnavailableError(RuntimeError):
    """The requested camera / file cannot be opened (missing hardware or package)."""


@dataclass
class CapturedFrame:
    """One frame plus everything we know about when it was captured."""

    index: int  # counter of delivered frames (0, 1, 2, ...)
    t_ns: int  # capture timestamp in nanoseconds (sensor, monotonic, video or synthetic clock)
    host_ns: int  # time.monotonic_ns() when the frame reached our code
    image: np.ndarray | None  # BGR uint8 image (H, W, 3); None for image-free synthetic frames
    timestamp_source: str  # "sensor" | "monotonic" | "video" | "synthetic"
    meta: dict[str, Any] = field(default_factory=dict)


class FrameSource(abc.ABC):
    """A camera-like object that delivers ``CapturedFrame`` objects in time order."""

    name: str = "base"
    is_synthetic: bool = False

    def __init__(self, width: int, height: int, fps: float) -> None:
        self.width = int(width)
        self.height = int(height)
        self.nominal_fps = float(fps)

    @abc.abstractmethod
    def open(self) -> None:
        """Start the source (open the camera / file)."""

    @abc.abstractmethod
    def read(self, timeout_s: float = 1.0) -> CapturedFrame | None:
        """Return the next frame, or None at the end of a file / on timeout."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the camera / file."""

    def describe(self) -> dict[str, Any]:
        """Settings to store in session.json (resolution, fps, locked exposure, ...)."""
        return {
            "source": self.name,
            "width": self.width,
            "height": self.height,
            "nominal_fps": self.nominal_fps,
            "synthetic": self.is_synthetic,
        }

    def on_trial_start(self, trial_info: dict[str, Any]) -> None:
        """Called by the recorder when a trial starts. Real cameras ignore it."""

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def monotonic_ns() -> int:
    """Monotonic clock in nanoseconds (never jumps backwards)."""
    return time.monotonic_ns()


def count_dropped_frames(dt_ns: int, nominal_fps: float, tolerance: float = 0.5) -> int:
    """How many frames are missing between two consecutive timestamps.

    With a nominal frame interval of 16.7 ms (60 fps), a gap of ~33 ms means
    one frame was dropped. ``tolerance`` = 0.5 means a gap must be at least
    1.5 intervals long before it counts as a drop, so normal jitter does not.
    """
    if nominal_fps <= 0:
        return 0
    interval_ns = 1e9 / nominal_fps
    missing = int(np.floor(dt_ns / interval_ns + tolerance)) - 1
    return max(missing, 0)
