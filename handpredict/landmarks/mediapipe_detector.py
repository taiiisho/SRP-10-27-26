"""MediaPipe Hand Landmarker on the CPU (works on laptops and on the Pi).

Uses the MediaPipe *Tasks* API (``mediapipe.tasks.python.vision.HandLandmarker``)
with the ``hand_landmarker.task`` model file (download it with
``handpredict download-models``). The older ``mp.solutions.hands`` API no
longer exists in MediaPipe 1.x.

The detector runs in VIDEO mode: after the hand is found once, MediaPipe
tracks it from frame to frame instead of searching the whole image again,
which is faster and steadier. VIDEO mode needs increasing timestamps, which
the camera timestamps provide.

"confidence" here is MediaPipe's handedness score (left/right hand
classification score); the Tasks API reports no per-landmark confidence for
hands.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..capture.base import CapturedFrame
from ..utils import sha256_file
from .base import DetectorUnavailableError, LandmarkDetector, LandmarkResult

log = logging.getLogger(__name__)


class MediaPipeHandDetector(LandmarkDetector):
    """CPU hand-landmark detector based on MediaPipe."""

    name = "mediapipe"
    hardware = "cpu"
    confidence_meaning = "MediaPipe handedness score (0-1)"

    def __init__(self, cfg: Mapping) -> None:
        super().__init__(cfg)
        self.mp_cfg = cfg["detector"]["mediapipe"]
        self.model_path = Path(self.mp_cfg["model_path"])
        # MediaPipe's C++ core logs a lot of harmless warnings; only show errors.
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import mediapipe as mp  # noqa: F401
            from mediapipe.tasks.python import BaseOptions, vision  # noqa: F401
        except ImportError as exc:
            raise DetectorUnavailableError(
                "mediapipe is not installed: pip install -e '.[mediapipe]'"
            ) from exc
        except OSError as exc:  # e.g. libEGL.so.1 missing on a minimal OS image
            raise DetectorUnavailableError(f"mediapipe could not load its native library: {exc}") from exc
        if not self.model_path.exists():
            raise DetectorUnavailableError(
                f"MediaPipe model {self.model_path} not found. Run `handpredict download-models`."
            )
        self.model_sha256 = sha256_file(self.model_path)
        self._landmarker = None
        self._last_ms = -1
        self._create()

    def _create(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(self.mp_cfg["num_hands"]),
            min_hand_detection_confidence=float(self.mp_cfg["min_hand_detection_confidence"]),
            min_hand_presence_confidence=float(self.mp_cfg["min_hand_presence_confidence"]),
            min_tracking_confidence=float(self.mp_cfg["min_tracking_confidence"]),
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except OSError as exc:
            raise DetectorUnavailableError(
                f"MediaPipe could not start ({exc}). On Raspberry Pi OS Lite install the EGL library: "
                "`sudo apt install libegl1 libgles2`."
            ) from exc
        self._last_ms = -1

    def reset(self) -> None:
        """Start a fresh tracker (the previous trial's hand position is forgotten)."""
        self.close()
        self._create()

    def _detect(self, frame: CapturedFrame) -> LandmarkResult:
        import cv2

        if frame.image is None:
            raise ValueError("MediaPipeHandDetector needs real images (the frame has none)")
        rgb = np.ascontiguousarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(frame.t_ns // 1_000_000)
        if timestamp_ms <= self._last_ms:  # VIDEO mode refuses non-increasing timestamps
            timestamp_ms = self._last_ms + 1
        self._last_ms = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return LandmarkResult(detected=False)
        h, w = frame.image.shape[:2]
        hand = result.hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in hand], dtype=float)
        lm = pts[self.landmark_index]
        score = result.handedness[0][0].score if result.handedness and result.handedness[0] else float("nan")
        hand_label = result.handedness[0][0].category_name if result.handedness and result.handedness[0] else None
        return LandmarkResult(True, lm[0], lm[1], lm[2], float(score), pts, extra={"handedness": hand_label})

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "model_path": str(self.model_path),
                "model_sha256": self.model_sha256,
                "settings": dict(self.mp_cfg),
            }
        )
        return info
