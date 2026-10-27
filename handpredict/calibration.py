"""Calibration: pixels -> centimetres, plus the start region and the target locations.

How the scale works: a measuring tape lies flat in the plane where the hand
moves. Two points on the tape a known distance apart (say 50 cm) are marked in
the camera image; their pixel distance gives centimetres per pixel. Marking
more than one tape segment (e.g. one horizontal, one vertical) shows whether
the scale is the same everywhere in the image: if the segments disagree by
more than a few percent, the camera is probably not square to the table.

Limitations (worth mentioning in the write-up): one scale factor assumes the
hand moves in a plane parallel to the image and ignores lens distortion. Both
are small for a camera pointed straight at the movement plane from ~1 m.

Targets: the minimum-jerk model is told where the hand will stop. What
matters is where the *tracked knuckle* ends up, not where the paper marker
is, so ``--capture-targets`` records the knuckle position while the hand is
held at each target, which avoids a systematic offset.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .utils import now_iso, read_json, write_json

log = logging.getLogger(__name__)


@dataclass
class ScaleSegment:
    """Two image points a known real distance apart."""

    p1_px: tuple[float, float]
    p2_px: tuple[float, float]
    distance_cm: float

    @property
    def length_px(self) -> float:
        return float(np.hypot(self.p2_px[0] - self.p1_px[0], self.p2_px[1] - self.p1_px[1]))

    @property
    def cm_per_px(self) -> float:
        if self.length_px <= 0:
            raise ValueError("the two calibration points must be different")
        return float(self.distance_cm) / self.length_px


@dataclass
class Region:
    """A circle in the image: the start region or a target."""

    center_px: tuple[float, float]
    radius_px: float
    method: str = "click"  # click | hand | manual

    def contains(self, xy_px) -> bool:
        return bool(np.hypot(xy_px[0] - self.center_px[0], xy_px[1] - self.center_px[1]) <= self.radius_px)


@dataclass
class Calibration:
    segments: list[ScaleSegment]
    image_size: tuple[int, int]
    start_region: Region | None = None
    targets: dict[str, Region] = field(default_factory=dict)
    created: str = field(default_factory=now_iso)
    notes: str = ""
    camera: dict[str, Any] = field(default_factory=dict)

    @property
    def cm_per_px(self) -> float:
        if not self.segments:
            raise ValueError("calibration has no scale segments")
        return float(np.mean([s.cm_per_px for s in self.segments]))

    @property
    def scale_spread(self) -> float:
        """(largest - smallest) / mean scale over the segments; 0 with a single segment."""
        values = [s.cm_per_px for s in self.segments]
        return float((max(values) - min(values)) / np.mean(values)) if len(values) > 1 else 0.0

    def px_to_cm(self, xy_px: np.ndarray) -> np.ndarray:
        return np.asarray(xy_px, dtype=float) * self.cm_per_px

    def target_px(self, target_id: str) -> np.ndarray:
        if target_id not in self.targets:
            raise KeyError(f"target {target_id!r} is not calibrated (known: {sorted(self.targets)})")
        return np.asarray(self.targets[target_id].center_px, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cm_per_px": self.cm_per_px,
            "scale_spread": self.scale_spread,
            "segments": [
                {"p1_px": list(s.p1_px), "p2_px": list(s.p2_px), "distance_cm": s.distance_cm,
                 "length_px": s.length_px, "cm_per_px": s.cm_per_px}
                for s in self.segments
            ],
            "image_size": list(self.image_size),
            "start_region": None if self.start_region is None else _region_dict(self.start_region),
            "targets": {k: _region_dict(v) for k, v in self.targets.items()},
            "created": self.created,
            "notes": self.notes,
            "camera": self.camera,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Calibration":
        segments = [ScaleSegment(tuple(s["p1_px"]), tuple(s["p2_px"]), float(s["distance_cm"])) for s in d["segments"]]
        start = d.get("start_region")
        return cls(
            segments=segments,
            image_size=tuple(d["image_size"]),
            start_region=None if start is None else _region_from(start),
            targets={k: _region_from(v) for k, v in d.get("targets", {}).items()},
            created=d.get("created", ""),
            notes=d.get("notes", ""),
            camera=d.get("camera", {}),
        )

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        return write_json(path, self.to_dict(), overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(read_json(path))

    def check(self, max_spread: float = 0.05) -> list[str]:
        """Plain-language warnings about a calibration."""
        warnings = []
        if self.scale_spread > max_spread:
            warnings.append(
                f"The tape segments disagree by {100 * self.scale_spread:.1f}% (more than {100 * max_spread:.0f}%): "
                "is the camera square to the movement plane?"
            )
        if self.start_region is None:
            warnings.append("No start region marked.")
        if not self.targets:
            warnings.append("No targets marked: the minimum-jerk model needs the target locations.")
        return warnings


def _region_dict(r: Region) -> dict[str, Any]:
    return {"center_px": list(r.center_px), "radius_px": r.radius_px, "method": r.method}


def _region_from(d: Mapping[str, Any]) -> Region:
    return Region(tuple(d["center_px"]), float(d["radius_px"]), d.get("method", "click"))


# -----------------------------------------------------------------------------
# Parsing command-line values (headless calibration over SSH)
# -----------------------------------------------------------------------------
def parse_numbers(text: str, count: int | Iterable[int]) -> list[float]:
    """Parse '320,400,960,410' into numbers, checking how many were given."""
    values = [float(v) for v in text.replace(" ", "").split(",") if v != ""]
    allowed = {count} if isinstance(count, int) else set(count)
    if len(values) not in allowed:
        raise ValueError(f"expected {sorted(allowed)} comma-separated numbers, got {text!r}")
    return values


def parse_target(text: str, default_radius: float) -> tuple[str, Region]:
    """Parse 'T1=750,160' or 'T1=750,160,40' into (target id, Region)."""
    if "=" not in text:
        raise ValueError(f"target must look like T1=x,y[,radius], got {text!r}")
    name, coords = text.split("=", 1)
    vals = parse_numbers(coords, (2, 3))
    radius = vals[2] if len(vals) == 3 else default_radius
    return name.strip(), Region((vals[0], vals[1]), radius, "manual")


def build_calibration(
    image_size: tuple[int, int],
    point_pairs: list[list[float]],
    distances_cm: list[float],
    start: list[float] | None,
    targets: dict[str, Region],
    default_start_radius: float,
    notes: str = "",
) -> Calibration:
    if len(point_pairs) != len(distances_cm):
        raise ValueError("give one --distance-cm for every --points pair")
    segments = [ScaleSegment((p[0], p[1]), (p[2], p[3]), d) for p, d in zip(point_pairs, distances_cm)]
    start_region = None
    if start is not None:
        radius = start[2] if len(start) == 3 else default_start_radius
        start_region = Region((start[0], start[1]), radius, "manual")
    return Calibration(segments, image_size, start_region, targets, notes=notes)


# -----------------------------------------------------------------------------
# Interactive clicking (needs a display)
# -----------------------------------------------------------------------------
def display_available() -> bool:
    """True if an OpenCV window can be shown (a desktop session, not plain SSH)."""
    if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import cv2

        return hasattr(cv2, "imshow") and "highgui" in cv2.getBuildInformation().lower()
    except Exception:  # pragma: no cover - depends on the OpenCV build
        return False


def click_points(image: np.ndarray, instructions: list[str], window: str = "calibration") -> list[tuple[float, float]]:
    """Show ``image`` and collect one mouse click per instruction (u = undo, Esc = cancel)."""
    import cv2

    clicks: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(instructions):
            clicks.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while len(clicks) < len(instructions):
            canvas = image.copy()
            for i, (x, y) in enumerate(clicks):
                cv2.circle(canvas, (int(x), int(y)), 6, (0, 255, 255), 2)
                cv2.putText(canvas, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            text = f"Click: {instructions[len(clicks)]}   (u = undo, Esc = cancel)"
            cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                raise KeyboardInterrupt("calibration cancelled")
            if key == ord("u") and clicks:
                clicks.pop()
    finally:
        cv2.destroyWindow(window)
    return clicks


# -----------------------------------------------------------------------------
# Measuring positions with the hand itself
# -----------------------------------------------------------------------------
def measure_hand_position(source, detector, hold_s: float, max_frames: int = 1000) -> tuple[np.ndarray, float, int]:
    """Median landmark position while the hand is held still for ``hold_s`` seconds.

    Returns (median xy in pixels, spread in pixels, number of detections).
    """
    first_ns = None
    points = []
    for _ in range(max_frames):
        frame = source.read()
        if frame is None:
            break
        if first_ns is None:
            first_ns = frame.t_ns
        result = detector.detect(frame)
        if result.detected:
            points.append((result.x_px, result.y_px))
        if (frame.t_ns - first_ns) / 1e9 >= hold_s:
            break
    if len(points) < 5:
        raise RuntimeError("The hand was not detected reliably; check lighting and that the hand is in view.")
    pts = np.asarray(points)
    median = np.median(pts, axis=0)
    spread = float(np.median(np.linalg.norm(pts - median, axis=1)))
    return median, spread, len(points)


def wait_for_operator(prompt: str) -> None:
    """Wait for Enter (works over SSH)."""
    input(prompt)
    time.sleep(0.05)
