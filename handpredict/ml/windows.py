"""Turning trials into (input, answer) examples for the ML models: sliding windows.

At every frame j that has a real detection and enough history, one example is
made:

* input "seq": the last W samples (default 200 ms of history) as positions
  RELATIVE to the current position, plus the time step Delta-t before each
  sample (frames are not perfectly evenly spaced);
* input "static": the current absolute position (switchable ablation) and the
  target position relative to the hand (the minimum-jerk model is told the
  target, so the ML model gets the same information; switchable ablation);
* answer: the displacement from the current position to the MEASURED
  position at every horizon (50 ... 500 ms) at once, with a mask for horizons
  that have no ground truth (end of recording / long gap).

Only samples at or before frame j enter the input, exactly like the physics
models. The same ``window_at`` function is used for training, evaluation and
the live demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..kinematics import ground_truth_at
from ..trial import TrialData


@dataclass(frozen=True)
class WindowSpec:
    n_samples: int  # W
    horizons_s: tuple[float, ...]
    absolute_position: bool
    target: str  # "relative" | "none"
    nominal_dt: float
    max_gap_s: float

    @classmethod
    def from_config(cls, cfg: Mapping, fps: float) -> "WindowSpec":
        ml = cfg["ml"]
        n = max(2, int(round(ml["history_ms"] / 1000.0 * fps)))
        target = ml["features"]["target"]
        if target not in ("relative", "none"):
            raise ValueError("ml.features.target must be 'relative' or 'none'")
        return cls(
            n_samples=n,
            horizons_s=tuple(float(h) / 1000.0 for h in cfg["horizons_ms"]),
            absolute_position=bool(ml["features"]["absolute_position"]),
            target=target,
            nominal_dt=1.0 / fps,
            max_gap_s=cfg["kinematics"]["ground_truth"]["max_gap_ms"] / 1000.0,
        )

    @property
    def n_horizons(self) -> int:
        return len(self.horizons_s)

    @property
    def n_static(self) -> int:
        return 2 * int(self.absolute_position) + 2 * int(self.target == "relative")

    @property
    def seq_features(self) -> int:
        return 3  # dx, dy, dt

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples, "horizons_s": list(self.horizons_s),
            "absolute_position": self.absolute_position, "target": self.target,
            "nominal_dt": self.nominal_dt, "max_gap_s": self.max_gap_s,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WindowSpec":
        return cls(int(d["n_samples"]), tuple(d["horizons_s"]), bool(d["absolute_position"]),
                   str(d["target"]), float(d["nominal_dt"]), float(d["max_gap_s"]))


def window_at(
    t: np.ndarray, xy: np.ndarray, measured: np.ndarray, j: int, spec: WindowSpec, target_xy: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Model input at frame j, built from samples 0..j only (None if not possible)."""
    w = spec.n_samples
    if j < w - 1 or not measured[j]:
        return None
    pts = xy[j - w + 1 : j + 1]
    if not np.all(np.isfinite(pts)):
        return None  # a long gap inside the window
    times = t[j - w + 1 : j + 1]
    dt = np.empty(w)
    dt[1:] = np.diff(times)
    dt[0] = t[j - w + 1] - t[j - w] if j - w >= 0 else spec.nominal_dt
    seq = np.column_stack([pts - xy[j], dt])
    static: list[float] = []
    if spec.absolute_position:
        static.extend(xy[j])
    if spec.target == "relative":
        if target_xy is None:
            return None
        static.extend(np.asarray(target_xy, float) - xy[j])
    return seq, np.asarray(static, dtype=float)


@dataclass
class WindowSet:
    """Many examples stacked: seq (N, W, 3), static (N, S), target (N, H, 2), mask (N, H)."""

    seq: np.ndarray
    static: np.ndarray
    target: np.ndarray
    mask: np.ndarray
    trial_id: np.ndarray
    frame_index: np.ndarray
    t_now: np.ndarray

    def __len__(self) -> int:
        return len(self.seq)

    def select(self, keep: np.ndarray) -> "WindowSet":
        return WindowSet(*(getattr(self, f)[keep] for f in _FIELDS))

    @staticmethod
    def concat(sets: Sequence["WindowSet"], spec: WindowSpec) -> "WindowSet":
        sets = [s for s in sets if len(s)]
        if not sets:
            return empty_windows(spec)
        return WindowSet(*(np.concatenate([getattr(s, f) for s in sets]) for f in _FIELDS))


_FIELDS = ("seq", "static", "target", "mask", "trial_id", "frame_index", "t_now")


def empty_windows(spec: WindowSpec) -> WindowSet:
    h, w, s = spec.n_horizons, spec.n_samples, spec.n_static
    return WindowSet(np.zeros((0, w, 3)), np.zeros((0, s)), np.zeros((0, h, 2)), np.zeros((0, h), bool),
                     np.zeros(0, dtype=object), np.zeros(0, int), np.zeros(0))


def build_windows(data: TrialData, spec: WindowSpec, frames: np.ndarray | None = None) -> WindowSet:
    """All possible examples of one trial (or only at the given frame indices)."""
    ctx = data.context
    candidates = range(data.n) if frames is None else frames
    seqs, statics, idx = [], [], []
    for j in candidates:
        built = window_at(data.t, data.xy, data.measured, int(j), spec, ctx.target_xy)
        if built is not None:
            seqs.append(built[0])
            statics.append(built[1])
            idx.append(int(j))
    if not idx:
        return empty_windows(spec)
    idx_arr = np.asarray(idx)
    horizons = np.asarray(spec.horizons_s)
    t_now = data.t[idx_arr]
    # Ground truth from REAL measurements only (no interpolated or filtered values).
    queries = (t_now[:, None] + horizons[None, :]).ravel()
    truth, valid = ground_truth_at(data.t[data.measured], data.xy[data.measured], queries, spec.max_gap_s)
    truth = truth.reshape(len(idx_arr), len(horizons), 2)
    valid = valid.reshape(len(idx_arr), len(horizons))
    target = truth - data.xy[idx_arr][:, None, :]
    target[~valid] = 0.0
    return WindowSet(
        seq=np.stack(seqs),
        static=np.stack(statics) if spec.n_static else np.zeros((len(idx), 0)),
        target=target,
        mask=valid,
        trial_id=np.array([ctx.trial_id] * len(idx), dtype=object),
        frame_index=idx_arr,
        t_now=t_now,
    )


def time_scaled_copy(data: TrialData, factor: float, rng: np.random.Generator) -> TrialData:
    """Augmentation: the same movement played back ``factor`` times faster (training data only).

    Positions are re-sampled from the real measurements at the original kind
    of frame times, so the copy looks like a slightly faster or slower trial
    recorded by the same camera.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    tm, xm = data.t[data.measured], data.xy[data.measured]
    if len(tm) < 2:
        return data
    duration = tm[-1] / factor
    dts = np.diff(data.t)
    step = float(np.median(dts)) if len(dts) else 1.0 / data.context.fps
    t_new = data.t.copy()
    if t_new[-1] < duration:
        extra = np.arange(t_new[-1] + step, duration + 1e-9, step)
        t_new = np.concatenate([t_new, extra])
    t_new = t_new[t_new <= duration + 1e-9]
    src = factor * t_new
    xy_new = np.column_stack([np.interp(src, tm, xm[:, k]) for k in range(2)])
    # Samples that would come from inside a long gap stay missing.
    right = np.clip(np.searchsorted(tm, src), 1, len(tm) - 1)
    gap = (tm[right] - tm[right - 1]) > 0.1
    measured = ~gap
    xy_new[gap] = np.nan
    return TrialData(data.context, t_new, xy_new, measured, np.zeros(len(t_new), bool))
