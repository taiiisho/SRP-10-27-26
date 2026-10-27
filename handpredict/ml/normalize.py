"""Input/output scaling for the networks, fitted on TRAINING windows only.

Neural networks learn best when their inputs and outputs are numbers of
roughly size 1. The scales are measured on the training data only; using the
test data here would leak information about it into the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .windows import WindowSet, WindowSpec


@dataclass
class Normalizer:
    pos_scale: float  # cm -> unit size for relative positions in the window
    dt_scale: float  # seconds -> unit size for the time steps
    abs_mean: np.ndarray  # (2,) mean absolute position
    abs_std: np.ndarray  # (2,) spread of absolute positions
    target_scale: float  # scale of the (target - current position) vector
    out_scale: np.ndarray  # (H,) scale of the answer at each horizon
    horizon_weights: np.ndarray  # (H,) loss weights (see the training settings)
    spec: WindowSpec

    @classmethod
    def fit(cls, windows: WindowSet, spec: WindowSpec, horizon_weighting: str = "inverse_baseline") -> "Normalizer":
        if len(windows) == 0:
            raise ValueError("cannot fit the normaliser on zero training windows")
        rel = windows.seq[:, :-1, :2].reshape(-1, 2)  # the last sample is always (0, 0)
        pos_scale = float(np.sqrt(np.mean(rel**2))) or 1.0
        s = 0
        abs_mean, abs_std = np.zeros(2), np.ones(2)
        if spec.absolute_position:
            abs_mean = windows.static[:, 0:2].mean(axis=0)
            abs_std = windows.static[:, 0:2].std(axis=0) + 1e-6
            s = 2
        target_scale = 1.0
        if spec.target == "relative":
            target_scale = float(np.sqrt(np.mean(windows.static[:, s : s + 2] ** 2))) or 1.0
        out_scale = np.ones(spec.n_horizons)
        baseline = np.ones(spec.n_horizons)
        for h in range(spec.n_horizons):
            valid = windows.mask[:, h]
            if valid.any():
                disp = windows.target[valid, h]
                out_scale[h] = max(float(np.sqrt(np.mean(disp**2))), 1e-3)
                baseline[h] = max(float(np.mean(np.linalg.norm(disp, axis=1))), 1e-3)
        if horizon_weighting == "inverse_baseline":
            # The zero-motion error grows with the horizon; dividing by it gives
            # every horizon a similar say in the training loss.
            weights = 1.0 / baseline
            weights = weights / weights.mean()
        elif horizon_weighting == "equal":
            weights = np.ones(spec.n_horizons)
        else:
            raise ValueError(f"unknown horizon_weighting {horizon_weighting!r}")
        return cls(pos_scale, spec.nominal_dt, abs_mean, abs_std, target_scale, out_scale, weights, spec)

    def inputs(self, seq: np.ndarray, static: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scale raw window inputs (cm, s) for the network (float32)."""
        seq_n = np.empty_like(seq, dtype=np.float32)
        seq_n[..., :2] = seq[..., :2] / self.pos_scale
        seq_n[..., 2] = seq[..., 2] / self.dt_scale
        static_n = np.empty_like(static, dtype=np.float32)
        s = 0
        if self.spec.absolute_position:
            static_n[:, 0:2] = (static[:, 0:2] - self.abs_mean) / self.abs_std
            s = 2
        if self.spec.target == "relative":
            static_n[:, s : s + 2] = static[:, s : s + 2] / self.target_scale
        return seq_n, static_n

    def outputs_to_cm(self, y_n: np.ndarray) -> np.ndarray:
        return np.asarray(y_n) * self.out_scale[None, :, None]

    def outputs_from_cm(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y) / self.out_scale[None, :, None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos_scale": self.pos_scale, "dt_scale": self.dt_scale,
            "abs_mean": self.abs_mean.tolist(), "abs_std": self.abs_std.tolist(),
            "target_scale": self.target_scale, "out_scale": self.out_scale.tolist(),
            "horizon_weights": self.horizon_weights.tolist(), "spec": self.spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Normalizer":
        return cls(
            float(d["pos_scale"]), float(d["dt_scale"]), np.asarray(d["abs_mean"], float), np.asarray(d["abs_std"], float),
            float(d["target_scale"]), np.asarray(d["out_scale"], float), np.asarray(d["horizon_weights"], float),
            WindowSpec.from_dict(d["spec"]),
        )
