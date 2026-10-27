"""The common interface of every hand-landmark detector (the measuring instrument).

A detector looks at one camera frame and returns the image position (pixels)
of ONE fixed hand landmark (by default MediaPipe landmark 9, the middle-finger
knuckle). Every detector also reports how long it took, so detection latency
can be logged separately from everything else.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..capture.base import CapturedFrame


class DetectorUnavailableError(RuntimeError):
    """The detector cannot run here (missing package, model file or accelerator)."""


@dataclass
class LandmarkResult:
    """What a detector found in one frame."""

    detected: bool
    x_px: float = float("nan")
    y_px: float = float("nan")
    z_rel: float = float("nan")  # MediaPipe's relative depth: logged, NOT used in the analysis
    confidence: float = float("nan")  # meaning depends on the backend (see detector.describe())
    landmarks_px: np.ndarray | None = None  # all 21 landmarks (x, y, z) for drawing / ROI tracking
    latency_ms: float = float("nan")  # time spent inside detect()
    extra: dict[str, Any] = field(default_factory=dict)


class LandmarkDetector(abc.ABC):
    """Base class: subclasses implement ``_detect``; ``detect`` adds the timing."""

    name: str = "base"
    hardware: str = "cpu"  # where the detection network runs: "cpu" or "hailo-10h"
    confidence_meaning: str = ""

    def __init__(self, cfg: Mapping) -> None:
        self.landmark_index = int(cfg["landmark"]["index"])
        self.landmark_name = cfg["landmark"]["name"]
        self.fallback_from: str | None = None  # set if this detector replaced an unavailable one

    def detect(self, frame: CapturedFrame) -> LandmarkResult:
        start = time.perf_counter()
        result = self._detect(frame)
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        return result

    @abc.abstractmethod
    def _detect(self, frame: CapturedFrame) -> LandmarkResult:
        ...

    def reset(self) -> None:
        """Forget tracking state (called at the start of every trial)."""

    def close(self) -> None:
        """Release resources (model, accelerator)."""

    def describe(self) -> dict[str, Any]:
        """Settings stored in session.json (backend, hardware, model file and checksum)."""
        return {
            "backend": self.name,
            "hardware": self.hardware,
            "landmark_index": self.landmark_index,
            "landmark_name": self.landmark_name,
            "confidence_meaning": self.confidence_meaning,
            "fallback_from": self.fallback_from,
        }

    def __enter__(self) -> "LandmarkDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
