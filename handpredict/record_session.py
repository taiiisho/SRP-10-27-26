"""Running a recording session (operator prompts) and replaying saved videos.

The operator workflow, which works over SSH without a screen:

    Trial 7/36  speed=fast  target=T2   hand detected in start region: yes
    [Enter] record   [l] change labels   [r] reject previous trial   [s] skip   [q] quit
    3... 2... 1...   (recording starts; participant holds still)
    GO!               (participant moves to the target)
    Trial 7 saved: 181 frames, 1.1 % missing -> OK

Every trial's speed and target come from a shuffled plan (each speed x target
combination repeated), so conditions are mixed rather than done in blocks.
"""

from __future__ import annotations

import logging
import sys
import time
import zlib
from pathlib import Path
from typing import Mapping

import numpy as np

from .calibration import Calibration, Region, ScaleSegment
from .capture import make_source
from .capture.video_source import VideoFileSource
from .landmarks import make_detector
from .rawdata import load_session, read_trial_table
from .recording import SessionWriter, TrialRecorder, build_trial_table, labels_dict, make_session_id
from .synthetic import SYNTHETIC_LABEL, trial_plan

log = logging.getLogger(__name__)


def synthetic_calibration(cfg: Mapping) -> Calibration:
    """The calibration that matches the synthetic source (exact scale, start and targets)."""
    syn = cfg["synthetic"]
    length_px = 50.0 / syn["cm_per_px"]
    cal_cfg = cfg["calibration"]
    targets = {k: Region(tuple(v), cal_cfg["default_target_radius_px"], "synthetic") for k, v in syn["targets_px"].items()}
    return Calibration(
        segments=[ScaleSegment((100.0, 100.0), (100.0 + length_px, 100.0), 50.0)],
        image_size=(cfg["camera"]["width"], cfg["camera"]["height"]),
        start_region=Region(tuple(syn["start_px"]), cal_cfg["default_start_radius_px"], "synthetic"),
        targets=targets,
        notes=f"{SYNTHETIC_LABEL} calibration generated from the config",
    )


def check_hand_in_start(source, detector, calibration: Calibration | None, n_frames: int = 8) -> str:
    """One-line status for the operator: is the hand visible and inside the start region?"""
    found = []
    for _ in range(n_frames):
        frame = source.read(timeout_s=1.0)
        if frame is None:
            break
        result = detector.detect(frame)
        if result.detected:
            found.append((result.x_px, result.y_px))
    if not found:
        return "hand NOT detected (check lighting / position)"
    xy = np.median(np.asarray(found), axis=0)
    if calibration is None or calibration.start_region is None:
        return f"hand detected at ({xy[0]:.0f}, {xy[1]:.0f}) px"
    inside = calibration.start_region.contains(xy)
    return f"hand detected at ({xy[0]:.0f}, {xy[1]:.0f}) px, in start region: {'yes' if inside else 'NO'}"


def _countdown(seconds: int) -> None:
    for k in range(seconds, 0, -1):
        log.info("%d...", k)
        time.sleep(1.0)


def _go_signal() -> None:
    sys.stderr.write("\a")  # terminal bell (audible on most terminals)
    log.info("GO!")


def run_recording_session(
    cfg: Mapping,
    participant_id: str,
    source_kind: str | None = None,
    detector_backend: str | None = None,
    calibration_path: str | Path | None = None,
    allow_uncalibrated: bool = False,
    video: str | None = None,
    n_trials: int | None = None,
    auto: bool = False,
    preview: bool = False,
    save_video: bool | None = None,
    notes: str = "",
    raw_dir: str | Path | None = None,
    session_id: str | None = None,
    seed: int | None = None,
) -> Path:
    """Record a full session; returns the session folder."""
    source_kind = source_kind or cfg["camera"]["source"]
    detector_backend = detector_backend or cfg["detector"]["backend"]
    if (source_kind == "synthetic") != (detector_backend == "synthetic"):
        raise ValueError("the synthetic source and the synthetic detector only work together")
    if calibration_path:
        calibration = Calibration.load(calibration_path)
    elif source_kind == "synthetic":
        calibration = synthetic_calibration(cfg)
    elif allow_uncalibrated:
        log.warning("Recording WITHOUT calibration: positions stay in pixels and no targets are known.")
        calibration = None
    else:
        raise ValueError("--calibration is required for real recordings (run `handpredict calibrate` first)")
    if calibration is not None:
        for warning in calibration.check():
            log.warning("Calibration: %s", warning)

    save_video = cfg["recording"]["save_video"] if save_video is None else save_video
    synthetic = source_kind == "synthetic"
    session_id = session_id or make_session_id(participant_id, synthetic)
    if seed is None:
        # Each session gets its own trial order (derived from the study seed
        # and the session id), so order effects do not line up with conditions
        # across sessions. The plan and its seed are stored in session.json.
        seed = (int(cfg["study"]["seed"]) ^ zlib.crc32(session_id.encode())) % (2**31)
    rng = np.random.default_rng(seed)
    proto = cfg["recording"]["protocol"]
    plan = trial_plan(proto["speeds"], proto["targets"], proto["repetitions"], rng)
    if n_trials is not None:
        plan = [plan[i % len(plan)] for i in range(n_trials)]

    source = make_source(cfg, source_kind, video_path=video, seed=seed)
    detector = make_detector(cfg, detector_backend)
    if calibration is not None and calibration.start_region is not None:
        detector.set_search_region(calibration.start_region.center_px)  # where trials begin (Hailo tracking)
    with source, detector:
        session = SessionWriter(
            cfg, participant_id, source, detector, calibration,
            raw_dir=raw_dir, session_id=session_id, notes=notes,
            extra={"trial_plan": [{"speed_condition": s, "target_id": t} for s, t in plan], "plan_seed": seed},
        )
        recorder = TrialRecorder(source, detector, cfg, calibration, preview=preview)
        last_trial_id: str | None = None
        queue_plan = list(plan)
        number = 0
        while queue_plan:
            speed, target = queue_plan[0]
            labels = labels_dict(speed, target)
            if not auto:
                status = check_hand_in_start(source, detector, calibration) if source.is_live else ""
                log.info("")
                log.info("Trial %d/%d  speed=%s  target=%s   %s", number + 1, number + len(queue_plan), speed, target, status)
                answer = input("[Enter] record  [l] change labels  [r] reject previous  [s] skip  [q] quit > ").strip().lower()
                if answer == "q":
                    break
                if answer == "s":
                    queue_plan.pop(0)
                    continue
                if answer == "r":
                    if last_trial_id is None:
                        log.warning("No previous trial to reject.")
                    else:
                        reason = input("Reason for rejecting the previous trial: ").strip()
                        session.flag_trial(last_trial_id, "operator_rejected", reason)
                        log.info("Previous trial flagged as rejected (its files are kept). Add a redo with the same labels.")
                        queue_plan.insert(0, (speed, target))
                    continue
                if answer == "l":
                    labels["speed_condition"] = input(f"speed [{speed}]: ").strip() or speed
                    labels["target_id"] = input(f"target [{target}]: ").strip() or target
                    labels["planned"] = False
                _countdown(int(cfg["recording"]["countdown_s"]))
            queue_plan.pop(0)
            number += 1
            stem = session.trial_stem(session.n_trials + 1)
            video_path = session.dir / f"{stem}.mp4" if save_video else None
            recording = recorder.record(
                {"speed_condition": labels["speed_condition"], "target_id": labels["target_id"]},
                go_callback=None if auto else _go_signal,
                video_path=video_path,
            )
            table, quality = build_trial_table(recording, cfg, calibration, session.data_origin)
            if not auto:
                labels["notes"] = input("Notes for this trial (Enter = none): ").strip()
            csv_path = session.save_trial(table, quality, labels, recording, video_path)
            last_trial_id = f"{session.session_id}/{stem}"
            verdict = "OK" if quality["quality_ok"] else "TOO MANY BAD FRAMES - consider redoing it ([r])"
            log.info(
                "Trial %d saved (%s): %d frames, %.1f %% bad, %d dropped, %.1f fps -> %s",
                number, csv_path.name, quality["n_frames"], 100 * quality["bad_fraction"],
                quality["dropped_frames"], quality["effective_fps"], verdict,
            )
        session.finish({"planned_trials": len(plan)})
    log.info("Session finished: %s (%d trials)", session.dir, session.n_trials)
    return session.dir


def replay_session(cfg: Mapping, session_dir: str | Path, detector_backend: str, raw_dir: str | Path | None = None) -> Path:
    """Re-run a detector on a session's saved videos and write the result as a NEW session.

    The original session is not touched. The new session records which
    session it was replayed from.
    """
    original = load_session(session_dir)
    trials = [t for t in original.trials if t.video_path is not None and t.video_path.exists()]
    if not trials:
        raise ValueError(f"{session_dir} has no saved videos (record with --save-video to allow replay)")
    detector = make_detector(cfg, detector_backend)
    first_source = VideoFileSource(trials[0].video_path, fps=original.fps)
    calibration = original.calibration
    if calibration is not None and calibration.start_region is not None:
        detector.set_search_region(calibration.start_region.center_px)
    session_id = f"{original.session_id}_replay_{detector.name}"
    with detector:
        session = SessionWriter(
            cfg, original.meta["participant_id"], first_source, detector, calibration,
            raw_dir=raw_dir or Path(session_dir).parent, session_id=session_id,
            notes=f"replay of {original.session_id}", synthetic=original.synthetic,
            extra={"replay_of": original.session_id, "original_camera": original.meta.get("camera")},
        )
        for trial in trials:
            table = read_trial_table(trial)
            source = VideoFileSource(trial.video_path, timestamps_ns=table["t_ns"].to_numpy(), fps=original.fps)
            with source:
                recorder = TrialRecorder(source, detector, cfg, calibration)
                recording = recorder.record(
                    {"speed_condition": trial.speed_condition, "target_id": trial.target_id}, limit_duration=False
                )
            # Keep the original GO flags (the replay itself has no GO signal).
            for rec, after_go in zip(recording.records, table["after_go"].to_numpy()):
                rec.after_go = bool(after_go)
            out, quality = build_trial_table(recording, cfg, calibration, session.data_origin)
            labels = labels_dict(trial.speed_condition, trial.target_id, f"replay of {trial.trial_id}", trial.info.get("planned", True))
            session.save_trial(out, quality, labels, recording)
            log.info("Replayed %s: %.1f %% bad frames", trial.trial_id, 100 * quality["bad_fraction"])
        session.finish({"replay_of": original.session_id})
    return session.dir
