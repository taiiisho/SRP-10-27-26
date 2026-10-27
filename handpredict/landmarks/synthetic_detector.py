"""A pretend detector for synthetic frames: true position + realistic detector errors.

For tests and ``selftest`` only. It reads the true hand position that the
synthetic source stored in each frame and adds Gaussian jitter, occasional
missed detections and rare outliers, like a real landmark detector.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..capture.base import CapturedFrame
from .base import LandmarkDetector, LandmarkResult


class SyntheticDetector(LandmarkDetector):
    name = "synthetic"
    hardware = "none (synthetic)"
    confidence_meaning = "constant 1.0 (synthetic)"

    def __init__(self, cfg: Mapping, seed: int | None = None) -> None:
        super().__init__(cfg)
        self.syn = cfg["detector"]["synthetic"]
        self.seed = (cfg["study"]["seed"] + 1) if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

    def _detect(self, frame: CapturedFrame) -> LandmarkResult:
        if "true_xy_px" not in frame.meta:
            raise ValueError("SyntheticDetector only works with frames from the synthetic source")
        if self.rng.random() < self.syn["miss_probability"]:
            return LandmarkResult(detected=False)
        xy = np.asarray(frame.meta["true_xy_px"], dtype=float) + self.rng.normal(0.0, self.syn["noise_px"], 2)
        if self.rng.random() < self.syn["outlier_probability"]:
            xy = xy + self.rng.normal(0.0, self.syn["outlier_px"], 2)
        return LandmarkResult(True, float(xy[0]), float(xy[1]), 0.0, 1.0)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({"seed": self.seed, "settings": dict(self.syn)})
        return info
