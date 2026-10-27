"""``handpredict demo``: live camera view with the physics and ML predictions drawn on top.

* green dot: the tracked knuckle now
* blue ring: where the physics model (minimum jerk) says it will be in h ms
* orange cross: where the ML model says it will be in h ms
* top line: horizon, rolling mean error of both models over the last few
  seconds (each prediction is scored when its moment arrives), frame rate

Keys (window mode): 1-7 choose the horizon, q or Esc quits. Over SSH without
a screen (headless mode) it prints the same numbers once per second and can
write the overlay to a video file (only if --output is given: the video
shows the participant).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Mapping

import numpy as np

from .calibration import Calibration, display_available
from .capture import make_source
from .landmarks import make_detector
from .online import OnlinePredictor
from .physics import make_physics_predictors
from .plotting import OKABE_ITO
from .trial import TrialContext

log = logging.getLogger(__name__)


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


class RollingError:
    """Scores predictions once their target time has passed (linear interpolation between frames)."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.pending: deque = deque()  # (target_time, model, horizon_index, predicted_xy)
        self.errors: dict[tuple[str, int], deque] = {}
        self.last: tuple[float, np.ndarray] | None = None

    def add(self, t_target: float, model: str, h_index: int, xy: np.ndarray) -> None:
        if np.all(np.isfinite(xy)):
            self.pending.append((t_target, model, h_index, np.asarray(xy, float)))

    def measure(self, t: float, xy: np.ndarray | None) -> None:
        if xy is None:
            return
        if self.last is not None:
            t0, xy0 = self.last
            still = deque()
            for item in self.pending:
                t_target, model, k, pred = item
                if t0 <= t_target <= t:
                    w = (t_target - t0) / (t - t0) if t > t0 else 0.0
                    truth = xy0 + w * (xy - xy0)
                    self.errors.setdefault((model, k), deque()).append((t, float(np.linalg.norm(pred - truth))))
                elif t_target > t:
                    still.append(item)
            self.pending = still
        self.last = (t, np.asarray(xy, float))
        for q in self.errors.values():
            while q and q[0][0] < t - self.window_s:
                q.popleft()

    def mean(self, model: str, h_index: int) -> float:
        q = self.errors.get((model, h_index))
        return float(np.mean([e for _, e in q])) if q else float("nan")


def run_demo(
    cfg: Mapping,
    source_kind: str | None = None,
    detector_backend: str | None = None,
    calibration_path: str | None = None,
    models_dir: str | None = None,
    horizon_ms: float | None = None,
    target_id: str | None = None,
    display: str | None = None,
    output_video: str | None = None,
    duration_s: float | None = None,
    video: str | None = None,
) -> dict:
    import cv2

    from .record_session import synthetic_calibration

    dcfg = cfg["demo"]
    source_kind = source_kind or cfg["camera"]["source"]
    detector_backend = detector_backend or cfg["detector"]["backend"]
    calibration = Calibration.load(calibration_path) if calibration_path else (
        synthetic_calibration(cfg) if source_kind == "synthetic" else None)
    if calibration is None:
        raise ValueError("the demo needs --calibration (centimetres and the target location)")
    target_id = target_id or dcfg["target_id"] or sorted(calibration.targets)[0]
    scale = calibration.cm_per_px
    context = TrialContext(
        "live", "demo", "-", "-", target_id,
        None if calibration.start_region is None else np.asarray(calibration.start_region.center_px) * scale,
        calibration.target_px(target_id) * scale, "cm", float(cfg["camera"]["fps"]), cm_per_px=scale,
    )
    physics_name = cfg["physics"]["headline"]
    predictors = {physics_name: make_physics_predictors(cfg)[physics_name]}
    ml_name = None
    if models_dir and Path(models_dir, "training_summary.json").exists():
        from .ml.inference import load_ml_predictors

        ml = load_ml_predictors(models_dir, include_seeds=False)
        ml_name = cfg["ml"]["headline"] if cfg["ml"]["headline"] in ml else next(iter(ml))
        predictors[ml_name] = ml[ml_name]
    else:
        log.warning("No trained models given (--models-dir): showing the physics prediction only")
    horizons_ms = [float(h) for h in cfg["horizons_ms"]]
    horizons = np.asarray(horizons_ms) / 1000.0
    h_index = horizons_ms.index(float(horizon_ms or dcfg["horizon_ms"]))
    mode = display or dcfg["display"]
    show = (mode == "window") or (mode == "auto" and display_available())
    source = make_source(cfg, source_kind, video_path=video, render=detector_backend != "synthetic")
    detector = make_detector(cfg, detector_backend)
    if calibration.start_region is not None:
        detector.set_search_region(calibration.start_region.center_px)
    online = OnlinePredictor(cfg, context, predictors)
    scorer = RollingError(float(dcfg["error_window_s"]))
    writer = None
    frames, first_t = 0, None
    fps_times: deque = deque(maxlen=30)
    start = time.perf_counter()
    last_log = start
    colours = {physics_name: _bgr(OKABE_ITO["blue"]), ml_name: _bgr(OKABE_ITO["vermillion"])}
    with source, detector:
        source.on_trial_start({"target_id": target_id, "speed_condition": "medium", "pre_roll_s": 0.5})
        while duration_s is None or time.perf_counter() - start < duration_s:
            frame = source.read(timeout_s=2.0)
            if frame is None:
                break
            frames += 1
            fps_times.append(time.perf_counter())
            first_t = frame.t_ns if first_t is None else first_t
            t = (frame.t_ns - first_t) / 1e9
            result = detector.detect(frame)
            xy_cm = np.array([result.x_px, result.y_px]) * scale if result.detected else None
            online.update(t, xy_cm)
            scorer.measure(t, xy_cm)
            preds = online.predict(horizons)
            for name, p in preds.items():
                for k, h in enumerate(horizons):
                    scorer.add(t + h, name, k, p.positions[k])
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) if len(fps_times) > 1 else 0.0
            errors = {name: scorer.mean(name, h_index) for name in predictors}
            status = f"horizon {horizons_ms[h_index]:g} ms | " + " | ".join(
                f"{name}: {err:.2f} cm" for name, err in errors.items()) + f" | {fps:.0f} fps | detect {result.latency_ms:.1f} ms"
            if frame.image is not None and (show or output_video):
                canvas = frame.image.copy()
                for region, colour in ((calibration.start_region, (200, 200, 200)), (calibration.targets.get(target_id), (0, 200, 255))):
                    if region is not None:
                        cv2.circle(canvas, tuple(int(v) for v in region.center_px), int(region.radius_px), colour, 2)
                if result.detected:
                    cv2.circle(canvas, (int(result.x_px), int(result.y_px)), 7, (0, 220, 0), -1)
                for name, p in preds.items():
                    pt = p.positions[h_index] / scale
                    if np.all(np.isfinite(pt)):
                        if name == physics_name:
                            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 10, colours[name], 2)
                        else:
                            cv2.drawMarker(canvas, (int(pt[0]), int(pt[1])), colours[name], cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)  # banner behind the text
                cv2.putText(canvas, status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                if output_video:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), source.nominal_fps, (w, h))
                    writer.write(canvas)
                if show:
                    cv2.imshow("handpredict demo", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if ord("1") <= key < ord("1") + len(horizons_ms):
                        h_index = key - ord("1")
            if time.perf_counter() - last_log >= 1.0:
                log.info(status)
                last_log = time.perf_counter()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    return {"frames": frames, "rolling_errors_cm": {name: scorer.mean(name, h_index) for name in predictors},
            "horizon_ms": horizons_ms[h_index]}
