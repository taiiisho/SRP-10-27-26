"""Recording trials: capture -> landmark detection -> raw files + session metadata.

Design choices (and why):

* **Two threads.** One thread only collects frames from the camera and puts
  them in a queue; the main thread runs the detector. A slow CPU detector
  therefore delays the results but does not make the camera skip frames
  (until the queue's memory cap is reached, which is counted). Frames only
  live in RAM; nothing visual is stored unless ``--save-video`` is given.
* **Real timestamps.** Each row keeps the frame's own timestamp. Frames the
  camera skipped show up as timestamp gaps and are counted per row
  (``dropped_before``).
* **Raw data is sacred.** Every raw file is written once, refused if it
  already exists, and made read-only. Later steps never edit it; operator
  decisions such as "reject this trial" go into an append-only flags file.
* **Missing detections.** Gaps of up to ``recording.max_gap_frames`` frames are
  linearly interpolated and flagged (``interpolated = 1``); longer gaps stay
  empty. A trial with too many bad frames is reported to the operator at once
  so it can be redone.
"""

from __future__ import annotations

import csv
import logging
import queue
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .calibration import Calibration
from .capture.base import CapturedFrame, FrameSource, count_dropped_frames, monotonic_ns
from .config import config_hash
from .kinematics import fill_short_gaps, make_filter
from .landmarks.base import LandmarkDetector
from .synthetic import SYNTHETIC_LABEL
from .utils import (
    RawDataExistsError,
    git_commit,
    hardware_description,
    make_read_only,
    now_iso,
    sha256_file,
    software_versions,
    write_json,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

TRIAL_COLUMNS = [
    "frame_index",       # 0, 1, 2, ... within the trial
    "t_ns",              # capture timestamp (camera sensor clock / monotonic clock), nanoseconds
    "t_s",               # seconds since the first frame of the trial
    "host_ns",           # monotonic time when the frame reached our code
    "dropped_before",    # frames the camera skipped just before this one (from timestamp gaps)
    "after_go",          # 1 once the "GO" signal has been given
    "detected",          # 1 = the detector found the hand in this frame
    "interpolated",      # 1 = short gap filled by linear interpolation (not a measurement)
    "x_px", "y_px",      # landmark position in pixels (image origin top-left, y downwards)
    "x_cm", "y_cm",      # the same position in centimetres (calibrated scale)
    "z_rel",             # MediaPipe relative depth: logged only, never used in the analysis
    "confidence",        # detector score (meaning in session.json -> detector.confidence_meaning)
    "detect_ms",         # time the detector needed for this frame
    "result_latency_ms", # frame arrival -> detection result (includes waiting in the queue)
    "data_origin",       # "real" or "synthetic"
]

INDEX_COLUMNS = [
    "trial_id", "stem", "speed_condition", "target_id", "n_frames", "n_detected", "n_interpolated",
    "n_missing", "dropped_frames", "queue_overflow", "bad_fraction", "quality_ok", "stopped_reason",
    "effective_fps", "mean_detect_ms", "p95_detect_ms", "recorded_at", "notes",
]

_PARTICIPANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def validate_participant_id(pid: str) -> str:
    """Participant IDs must be pseudonyms like P01 (letters, digits, '-' and '_' only)."""
    if not _PARTICIPANT_RE.match(pid or ""):
        raise ValueError(
            f"participant id {pid!r} is not allowed: use a pseudonym such as P01 "
            "(letters, digits, '-' or '_', no spaces, never a real name)"
        )
    return pid


# -----------------------------------------------------------------------------
# One recorded trial
# -----------------------------------------------------------------------------
@dataclass
class FrameRecord:
    frame_index: int
    t_ns: int
    host_ns: int
    dropped_before: int
    after_go: bool
    detected: bool
    x_px: float
    y_px: float
    z_rel: float
    confidence: float
    detect_ms: float
    result_latency_ms: float


@dataclass
class TrialRecording:
    records: list[FrameRecord]
    queue_overflow: int
    stopped_reason: str
    go_t_ns: int | None
    extra: dict[str, Any] = field(default_factory=dict)


class SettleDetector:
    """Online check 'the hand moved and has now come to rest (at the target)'."""

    def __init__(self, cfg: Mapping, cm_per_px: float, target_px: np.ndarray | None, target_radius_px: float | None):
        rec = cfg["recording"]
        self.stepper = make_filter(cfg).new_state()
        self.cm_per_px = cm_per_px
        self.onset = cfg["kinematics"]["segmentation"]["onset_speed_cm_s"]
        self.settle_speed = rec["settle_speed_cm_s"]
        self.settle_time = rec["settle_time_s"]
        self.target_px = target_px
        self.target_radius_px = target_radius_px
        self.moved = False
        self.still_since: float | None = None
        self.settled_at: float | None = None  # time (s since trial start) when the hand came to rest

    def update(self, t_s: float, xy_px: np.ndarray | None, after_go: bool) -> bool:
        """Feed one frame; returns True once the hand has settled (and stays True)."""
        if self.settled_at is not None:
            return True
        z = np.full(2, np.nan) if xy_px is None else np.asarray(xy_px, float) * self.cm_per_px
        _, v, _, _ = self.stepper.update(t_s, z, xy_px is not None)
        speed = float(np.hypot(*v)) if np.all(np.isfinite(v)) else 0.0
        if not self.moved:
            self.moved = after_go and speed > self.onset
            return False
        at_target = True
        if self.target_px is not None and xy_px is not None and self.target_radius_px:
            at_target = float(np.hypot(*(np.asarray(xy_px) - self.target_px))) <= self.target_radius_px
        if speed < self.settle_speed and at_target:
            self.still_since = t_s if self.still_since is None else self.still_since
            if t_s - self.still_since >= self.settle_time:
                self.settled_at = t_s
                return True
            return False
        self.still_since = None
        return False


class TrialRecorder:
    """Records one trial: reads frames, detects the landmark, returns per-frame records."""

    def __init__(
        self,
        source: FrameSource,
        detector: LandmarkDetector,
        cfg: Mapping,
        calibration: Calibration | None = None,
        preview: bool = False,
    ) -> None:
        self.source = source
        self.detector = detector
        self.cfg = cfg
        self.calibration = calibration
        self.preview = preview

    def record(
        self,
        trial_info: Mapping[str, Any],
        go_callback: Callable[[], None] | None = None,
        video_path: Path | None = None,
        limit_duration: bool = True,
    ) -> TrialRecording:
        rec_cfg = self.cfg["recording"]
        pre_roll = float(rec_cfg["pre_roll_s"])
        settle_mode = bool(rec_cfg["stop_when_settled"]) and self.calibration is not None
        hard_limit = float(rec_cfg["max_duration_s"]) if settle_mode else pre_roll + float(rec_cfg["duration_s"])
        live = bool(getattr(self.source, "is_live", False))
        frames: queue.Queue[CapturedFrame | None] = queue.Queue(maxsize=int(rec_cfg["max_queue_frames"]))
        stop = threading.Event()
        state: dict[str, Any] = {"overflow": 0, "go_t_ns": None, "reason": None, "error": None}

        self.detector.reset()
        self.source.on_trial_start({**trial_info, "pre_roll_s": pre_roll})

        def capture_loop() -> None:
            first_t = None
            try:
                while not stop.is_set():
                    frame = self.source.read(timeout_s=1.0)
                    if frame is None:
                        state["reason"] = state["reason"] or "end_of_stream"
                        break
                    first_t = frame.t_ns if first_t is None else first_t
                    elapsed = (frame.t_ns - first_t) / 1e9
                    if limit_duration and elapsed > hard_limit + 1e-6:
                        state["reason"] = state["reason"] or ("max_duration" if settle_mode else "duration")
                        break
                    frame.meta["after_go"] = elapsed >= pre_roll
                    if frame.meta["after_go"] and state["go_t_ns"] is None:
                        state["go_t_ns"] = frame.t_ns
                        if go_callback is not None:
                            go_callback()
                    if live:
                        try:
                            frames.put_nowait(frame)
                        except queue.Full:
                            state["overflow"] += 1  # shows up as a timestamp gap, i.e. a dropped frame
                    else:
                        frames.put(frame)  # files / synthetic: wait instead of losing frames
            except Exception as exc:  # pragma: no cover - hardware errors
                state["error"] = exc
            finally:
                frames.put(None)

        thread = threading.Thread(target=capture_loop, name="capture", daemon=True)
        thread.start()

        settle = None
        if settle_mode:
            target = trial_info.get("target_id")
            region = self.calibration.targets.get(target) if target else None
            settle = SettleDetector(
                self.cfg, self.calibration.cm_per_px,
                None if region is None else np.asarray(region.center_px), None if region is None else region.radius_px,
            )
        writer = None
        records: list[FrameRecord] = []
        prev_t: int | None = None
        first_t: int | None = None
        try:
            while True:
                frame = frames.get()
                if frame is None:
                    break
                result = self.detector.detect(frame)
                done_ns = monotonic_ns()
                first_t = frame.t_ns if first_t is None else first_t
                dropped = 0 if prev_t is None else count_dropped_frames(frame.t_ns - prev_t, self.source.nominal_fps)
                prev_t = frame.t_ns
                records.append(
                    FrameRecord(
                        frame_index=len(records),
                        t_ns=int(frame.t_ns),
                        host_ns=int(frame.host_ns),
                        dropped_before=dropped,
                        after_go=bool(frame.meta.get("after_go", False)),
                        detected=bool(result.detected),
                        x_px=result.x_px,
                        y_px=result.y_px,
                        z_rel=result.z_rel,
                        confidence=result.confidence,
                        detect_ms=result.latency_ms,
                        result_latency_ms=(done_ns - frame.host_ns) / 1e6,
                    )
                )
                if video_path is not None and frame.image is not None:
                    writer = writer or _open_video_writer(video_path, frame.image, self.source.nominal_fps, self.cfg)
                    writer.write(frame.image)
                if self.preview and frame.image is not None and len(records) % 2 == 0:
                    _show_preview(frame.image, result, self.calibration, trial_info)
                if settle is not None:
                    xy = (result.x_px, result.y_px) if result.detected else None
                    t_rel = (frame.t_ns - first_t) / 1e9
                    # Keep recording a little after the hand settles, so that
                    # predictions made near the end still have ground truth.
                    if settle.update(t_rel, xy, frame.meta.get("after_go", False)) and (
                        t_rel - settle.settled_at >= float(rec_cfg["post_settle_s"])
                    ):
                        state["reason"] = "settled"
                        stop.set()
                        break  # frames queued after this moment are not part of the trial
        finally:
            stop.set()
            _drain_until_sentinel(frames, thread)
            thread.join(timeout=5.0)
            if writer is not None:
                writer.release()
        if state["error"] is not None:
            raise state["error"]
        extra = {}
        spec = getattr(self.source, "current_spec", None)
        if spec is not None:
            extra["synthetic_reach_spec"] = spec.as_dict()
        return TrialRecording(records, int(state["overflow"]), state["reason"] or "duration", state["go_t_ns"], extra)


def _drain_until_sentinel(frames: queue.Queue, thread: threading.Thread, timeout_s: float = 5.0) -> None:
    """Throw away frames still in the queue so the capture thread can finish."""
    while thread.is_alive() or not frames.empty():
        try:
            if frames.get(timeout=timeout_s) is None:
                return
        except queue.Empty:
            return


def _open_video_writer(path: Path, image: np.ndarray, fps: float, cfg: Mapping):
    import cv2

    if path.exists():
        raise RawDataExistsError(f"{path} already exists; refusing to overwrite it")
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cfg["recording"]["video_codec"]), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    return writer


def _show_preview(image, result, calibration, trial_info) -> None:  # pragma: no cover - needs a display
    import cv2

    canvas = image.copy()
    if calibration is not None:
        if calibration.start_region is not None:
            c = calibration.start_region
            cv2.circle(canvas, tuple(int(v) for v in c.center_px), int(c.radius_px), (255, 200, 0), 2)
        target = calibration.targets.get(trial_info.get("target_id", ""))
        if target is not None:
            cv2.circle(canvas, tuple(int(v) for v in target.center_px), int(target.radius_px), (0, 200, 255), 2)
    if result.detected:
        cv2.circle(canvas, (int(result.x_px), int(result.y_px)), 8, (0, 255, 0), -1)
    cv2.imshow("handpredict preview", canvas)
    cv2.waitKey(1)


# -----------------------------------------------------------------------------
# From records to the raw trial table
# -----------------------------------------------------------------------------
def build_trial_table(
    recording: TrialRecording, cfg: Mapping, calibration: Calibration | None, data_origin: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Turn frame records into the raw trial table (with gap filling) and quality numbers."""
    recs = recording.records
    if not recs:
        raise RuntimeError("no frames were recorded; is the camera running?")
    t_ns = np.array([r.t_ns for r in recs], dtype=np.int64)
    t_s = (t_ns - t_ns[0]) / 1e9
    detected = np.array([r.detected for r in recs], dtype=bool)
    xy = np.array([[r.x_px, r.y_px] for r in recs], dtype=float)
    filled, interpolated = fill_short_gaps(t_s, xy, detected, int(cfg["recording"]["max_gap_frames"]))
    scale = calibration.cm_per_px if calibration is not None else np.nan
    df = pd.DataFrame(
        {
            "frame_index": np.arange(len(recs)),
            "t_ns": t_ns,
            "t_s": np.round(t_s, 6),
            "host_ns": np.array([r.host_ns for r in recs], dtype=np.int64),
            "dropped_before": np.array([r.dropped_before for r in recs], dtype=int),
            "after_go": np.array([r.after_go for r in recs], dtype=int),
            "detected": detected.astype(int),
            "interpolated": interpolated.astype(int),
            "x_px": np.round(filled[:, 0], 3),
            "y_px": np.round(filled[:, 1], 3),
            "x_cm": np.round(filled[:, 0] * scale, 4),
            "y_cm": np.round(filled[:, 1] * scale, 4),
            "z_rel": np.round([r.z_rel for r in recs], 4),
            "confidence": np.round([r.confidence for r in recs], 4),
            "detect_ms": np.round([r.detect_ms for r in recs], 3),
            "result_latency_ms": np.round([r.result_latency_ms for r in recs], 3),
            "data_origin": data_origin,
        },
        columns=TRIAL_COLUMNS,
    )
    n = len(df)
    dropped = int(df["dropped_before"].sum())
    n_detected = int(detected.sum())
    n_interp = int(interpolated.sum())
    bad_fraction = (n - n_detected + dropped) / (n + dropped)
    duration = float(t_s[-1]) if n > 1 else 0.0
    detect_ms = df["detect_ms"].to_numpy(dtype=float)
    quality = {
        "n_frames": n,
        "n_detected": n_detected,
        "n_interpolated": n_interp,
        "n_missing": n - n_detected - n_interp,
        "dropped_frames": dropped,
        "queue_overflow": recording.queue_overflow,
        "bad_fraction": round(float(bad_fraction), 4),
        "quality_ok": bool(bad_fraction <= cfg["recording"]["max_bad_frame_fraction"]),
        "stopped_reason": recording.stopped_reason,
        "effective_fps": round((n - 1) / duration, 2) if duration > 0 else float("nan"),
        "mean_detect_ms": round(float(np.nanmean(detect_ms)), 3),
        "p95_detect_ms": round(float(np.nanpercentile(detect_ms, 95)), 3),
    }
    return df, quality


# -----------------------------------------------------------------------------
# Session folder
# -----------------------------------------------------------------------------
def make_session_id(participant_id: str, synthetic: bool, stamp: str | None = None) -> str:
    from datetime import datetime

    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = f"S{stamp}_{participant_id}"
    return f"{SYNTHETIC_LABEL}_{sid}" if synthetic else sid


class SessionWriter:
    """Creates a new session folder and writes its raw files (never overwriting anything)."""

    def __init__(
        self,
        cfg: Mapping,
        participant_id: str,
        source: FrameSource,
        detector: LandmarkDetector,
        calibration: Calibration | None,
        raw_dir: str | Path | None = None,
        session_id: str | None = None,
        notes: str = "",
        extra: Mapping[str, Any] | None = None,
        synthetic: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.participant_id = validate_participant_id(participant_id)
        if synthetic is None:
            synthetic = bool(source.is_synthetic) or detector.name == "synthetic"
        self.synthetic = bool(synthetic)
        self.data_origin = SYNTHETIC_LABEL if self.synthetic else "real"
        self.session_id = session_id or make_session_id(participant_id, self.synthetic)
        if self.synthetic and SYNTHETIC_LABEL not in self.session_id:
            self.session_id = f"{SYNTHETIC_LABEL}_{self.session_id}"
        root = Path(raw_dir or cfg["paths"]["raw_dir"])
        self.dir = root / self.session_id
        if self.dir.exists():
            raise RawDataExistsError(f"session folder {self.dir} already exists; refusing to overwrite it")
        self.dir.mkdir(parents=True)
        self.n_trials = 0
        meta = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created": now_iso(),
            "data_origin": self.data_origin,
            "synthetic": self.synthetic,
            "participant_id": self.participant_id,
            "notes": notes,
            "camera": source.describe(),
            "camera_pose": dict(cfg["camera"].get("pose", {})),
            "detector": detector.describe(),
            "landmark": dict(cfg["landmark"]),
            "calibration": None if calibration is None else calibration.to_dict(),
            "recording": dict(cfg["recording"]),
            "units": {
                "time": "t_s = seconds since the first frame of the trial (t_ns = raw timestamp)",
                "position": "x_px/y_px = image pixels (origin top-left, y down); x_cm/y_cm = pixels * cm_per_px",
            },
            "software": software_versions(),
            "git_commit": git_commit(),
            "hardware": hardware_description(),
            "config": cfg,
            "config_hash": config_hash(cfg),
        }
        if extra:
            meta.update(extra)
        path = write_json(self.dir / "session.json", meta)
        make_read_only(path)
        self.meta = meta
        self.index_path = self.dir / "trials_index.csv"
        with open(self.index_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(INDEX_COLUMNS)
        self.flags_path = self.dir / "trial_flags.csv"
        with open(self.flags_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["trial_id", "flag", "reason", "flagged_at"])
        log.info("New session %s (%s data) in %s", self.session_id, self.data_origin, self.dir)

    def trial_stem(self, number: int) -> str:
        stem = f"trial_{number:03d}"
        return f"{stem}_{SYNTHETIC_LABEL}" if self.synthetic else stem

    def save_trial(
        self,
        table: pd.DataFrame,
        quality: Mapping[str, Any],
        labels: Mapping[str, Any],
        recording: TrialRecording,
        video_path: Path | None = None,
    ) -> Path:
        self.n_trials += 1
        stem = self.trial_stem(self.n_trials)
        trial_id = f"{self.session_id}/{stem}"
        csv_path = self.dir / f"{stem}.csv"
        json_path = self.dir / f"{stem}.json"
        for p in (csv_path, json_path):
            if p.exists():
                raise RawDataExistsError(f"{p} already exists; refusing to overwrite it")
        table.to_csv(csv_path, index=False)
        make_read_only(csv_path)
        info = {
            "trial_id": trial_id,
            "stem": stem,
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "data_origin": self.data_origin,
            "speed_condition": labels["speed_condition"],
            "target_id": labels["target_id"],
            "notes": labels.get("notes", ""),
            "planned": labels.get("planned", True),
            "recorded_at": now_iso(),
            "quality": dict(quality),
            "go_t_ns": recording.go_t_ns,
            "csv_sha256": sha256_file(csv_path),
            "video": None,
            **recording.extra,
        }
        if video_path is not None and video_path.exists():
            make_read_only(video_path)
            info["video"] = {"file": video_path.name, "sha256": sha256_file(video_path)}
        write_json(json_path, info)
        make_read_only(json_path)
        row = {**{k: quality.get(k) for k in INDEX_COLUMNS}, "trial_id": trial_id, "stem": stem,
               "speed_condition": labels["speed_condition"], "target_id": labels["target_id"],
               "recorded_at": info["recorded_at"], "notes": labels.get("notes", "")}
        with open(self.index_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([row[k] for k in INDEX_COLUMNS])
        return csv_path

    def flag_trial(self, trial_id: str, flag: str, reason: str = "") -> None:
        """Append a flag (e.g. operator_rejected) without touching the trial's files."""
        with open(self.flags_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([trial_id, flag, reason, now_iso()])

    def finish(self, summary: Mapping[str, Any] | None = None) -> Path:
        info = {"session_id": self.session_id, "finished": now_iso(), "n_trials": self.n_trials, **(summary or {})}
        path = write_json(self.dir / "session_end.json", info)
        for p in (path, self.index_path, self.flags_path):
            make_read_only(p)
        return path


def labels_dict(speed_condition: str, target_id: str, notes: str = "", planned: bool = True) -> dict[str, Any]:
    return {"speed_condition": speed_condition, "target_id": target_id, "notes": notes, "planned": planned}
