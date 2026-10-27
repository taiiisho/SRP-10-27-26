"""Hand-landmark detectors: MediaPipe (CPU), Hailo (AI HAT+ 2) and synthetic (tests)."""

from __future__ import annotations

import logging
from typing import Mapping

from .base import DetectorUnavailableError, LandmarkDetector, LandmarkResult

log = logging.getLogger(__name__)


def make_detector(cfg: Mapping, backend: str | None = None) -> LandmarkDetector:
    """Create the configured detector.

    If the Hailo accelerator is requested but not usable and
    ``detector.fallback_to_cpu`` is true, MediaPipe on the CPU is used instead,
    a warning is logged, and the detector records ``fallback_from="hailo"`` so
    session.json shows which hardware really did the detection.
    """
    backend = backend or cfg["detector"]["backend"]
    if backend == "synthetic":
        from .synthetic_detector import SyntheticDetector

        return SyntheticDetector(cfg)
    if backend == "mediapipe":
        from .mediapipe_detector import MediaPipeHandDetector

        return MediaPipeHandDetector(cfg)
    if backend == "hailo":
        try:
            from .hailo_detector import HailoHandLandmarkDetector

            return HailoHandLandmarkDetector(cfg)
        except (DetectorUnavailableError, ImportError) as exc:
            if not cfg["detector"]["fallback_to_cpu"]:
                raise DetectorUnavailableError(f"Hailo detector unavailable: {exc}") from exc
            log.warning("Hailo device not found; falling back to CPU detector (MediaPipe). Reason: %s", exc)
            from .mediapipe_detector import MediaPipeHandDetector

            detector = MediaPipeHandDetector(cfg)
            detector.fallback_from = "hailo"
            return detector
    raise ValueError(f"Unknown detector backend {backend!r} (use mediapipe, hailo or synthetic)")


__all__ = ["DetectorUnavailableError", "LandmarkDetector", "LandmarkResult", "make_detector"]
