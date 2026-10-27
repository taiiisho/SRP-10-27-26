"""Command-line interface: ``handpredict <command> [options]``.

Run ``handpredict --help`` for the list of commands and
``handpredict <command> --help`` for the options of one command. Every
command accepts ``--config file.yaml`` and ``--set section.key=value``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from . import __version__
from .config import ConfigError, load_config
from .logging_setup import setup_logging

log = logging.getLogger("handpredict")

COMMANDS: dict[str, tuple[Callable, str]] = {}


def command(name: str, help_text: str):
    """Register a sub-command handler."""

    def decorate(func: Callable) -> Callable:
        COMMANDS[name] = (func, help_text)
        return func

    return decorate


# -----------------------------------------------------------------------------
# download-models
# -----------------------------------------------------------------------------
def _download(url: str, dest: Path, force: bool = False) -> Path:
    from .utils import sha256_file

    if dest.exists() and not force:
        log.info("%s already present (sha256 %s)", dest, sha256_file(dest)[:16])
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise RuntimeError(f"downloaded file from {url} is empty")
    tmp.replace(dest)
    log.info("Saved %s (%.1f MB, sha256 %s)", dest, dest.stat().st_size / 1e6, sha256_file(dest)[:16])
    return dest


@command("download-models", "download the MediaPipe and Hailo hand-landmark model files")
def cmd_download_models(args, cfg) -> int:
    det = cfg["detector"]
    wanted = args.which or ["mediapipe", "hailo"]
    if "mediapipe" in wanted:
        _download(det["mediapipe"]["model_url"], Path(det["mediapipe"]["model_path"]), args.force)
    if "hailo" in wanted:
        _download(det["hailo"]["hef_url"], Path(det["hailo"]["hef_path"]), args.force)
    if "test-assets" in wanted:
        # A public hand photo from MediaPipe's own test assets, used by the
        # optional real-inference tests (tests/test_mediapipe.py).
        dest = Path(cfg["paths"]["detector_models_dir"]) / "test_assets" / "thumb_up.jpg"
        _download(TEST_IMAGE_URL, dest, args.force)
    return 0


TEST_IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/thumb_up.jpg"


def _add_download_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "which", nargs="*", choices=["mediapipe", "hailo", "test-assets"],
        help="which files (default: mediapipe and hailo; test-assets = a hand photo for the tests)",
    )
    p.add_argument("--force", action="store_true", help="download again even if the file exists")


# -----------------------------------------------------------------------------
# check-hardware
# -----------------------------------------------------------------------------
@command("check-hardware", "report camera, Hailo accelerator, detector and Python environment status")
def cmd_check_hardware(args, cfg) -> int:
    from .utils import hardware_description, software_versions

    log.info("Hardware: %s", hardware_description())
    log.info("Software: %s", software_versions())
    temp = Path("/sys/class/thermal/thermal_zone0/temp")
    if temp.exists():
        log.info("CPU temperature: %.1f C", int(temp.read_text().strip()) / 1000)
    # Raspberry Pi camera
    try:
        from picamera2 import Picamera2  # type: ignore[import-not-found]

        cams = Picamera2.global_camera_info()
        log.info("Raspberry Pi cameras: %s", cams or "none detected")
    except Exception as exc:  # noqa: BLE001 - report anything
        log.info("Raspberry Pi camera: picamera2 not usable (%s)", exc)
    video_devices = sorted(str(p) for p in Path("/dev").glob("video*"))
    log.info("Video devices: %s", video_devices or "none")
    # Hailo
    if shutil.which("hailortcli"):
        out = subprocess.run(["hailortcli", "fw-control", "identify"], capture_output=True, text=True, timeout=20)
        log.info("hailortcli fw-control identify:\n%s", (out.stdout or out.stderr).strip())
    else:
        log.info("hailortcli not found: Hailo software not installed (on the Pi: sudo apt install hailo-h10-all)")
    try:
        import hailo_platform  # type: ignore[import-not-found]

        log.info("hailo_platform (PyHailoRT) importable, version %s", getattr(hailo_platform, "__version__", "?"))
        if args.hef:
            from .landmarks.hailo_detector import describe_hef

            log.info("HEF %s:\n%s", args.hef, describe_hef(args.hef))
    except ImportError:
        log.info("hailo_platform not importable (expected on a laptop)")
    # MediaPipe
    try:
        from .landmarks import make_detector

        det = make_detector(cfg, "mediapipe")
        log.info("MediaPipe detector OK (model %s)", det.model_path)
        det.close()
    except Exception as exc:  # noqa: BLE001
        log.info("MediaPipe detector not usable: %s", exc)
    try:
        import onnxruntime as ort

        log.info("ONNX Runtime %s providers: %s", ort.__version__, ort.get_available_providers())
    except ImportError:
        log.info("onnxruntime not installed")
    return 0


def _add_check_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hef", help="also print the input/output layers of this Hailo HEF file")


# -----------------------------------------------------------------------------
# calibrate
# -----------------------------------------------------------------------------
@command("calibrate", "measure cm-per-pixel and mark the start region and targets")
def cmd_calibrate(args, cfg) -> int:
    import cv2
    import numpy as np

    from .calibration import (
        Calibration,
        Region,
        ScaleSegment,
        click_points,
        display_available,
        measure_hand_position,
        parse_numbers,
        parse_target,
        wait_for_operator,
    )
    from .capture import make_source

    cal_cfg = cfg["calibration"]
    source = None
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(args.image)
    else:
        source = make_source(cfg, args.source, video_path=args.video)
        source.open()
        frame = None
        for _ in range(max(1, int(cfg["camera"]["fps"] * 0.5))):  # let the camera settle
            frame = source.read() or frame
        if frame is None or frame.image is None:
            raise RuntimeError("could not grab a calibration frame from the camera")
        image = frame.image
    h, w = image.shape[:2]
    interactive = display_available() and not args.points
    if args.save_frame:
        cv2.imwrite(args.save_frame, image)
        log.info("Calibration frame saved to %s (make sure no person is visible in it)", args.save_frame)

    # 1) scale
    if args.points:
        pairs = [parse_numbers(p, 4) for p in args.points]
    elif interactive:
        pairs = []
        for k in range(args.segments):
            (x1, y1), (x2, y2) = click_points(image, [f"segment {k + 1}: first tape mark", f"segment {k + 1}: second tape mark"])
            pairs.append([x1, y1, x2, y2])
    else:
        path = args.save_frame or str(Path(args.out).with_name("calibration_frame.png"))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        log.error(
            "No display available. The frame was saved to %s: open it on another computer, read the pixel "
            "coordinates of two tape marks and run again with --points X1,Y1,X2,Y2 --distance-cm D", path,
        )
        return 2
    distances = list(args.distance_cm) if args.distance_cm else []
    while len(distances) < len(pairs):
        distances.append(float(input(f"Real distance between the marks of segment {len(distances) + 1} (cm): ")))
    segments = [ScaleSegment((p[0], p[1]), (p[2], p[3]), d) for p, d in zip(pairs, distances)]
    calibration = Calibration(segments, (w, h), notes=args.notes or "")
    if source is not None:
        calibration.camera = source.describe()

    # 2) start region and targets
    detector = None
    if args.capture_start or args.capture_targets:
        from .landmarks import make_detector

        detector = make_detector(cfg, args.detector)
    if args.start:
        vals = parse_numbers(args.start, (2, 3))
        calibration.start_region = Region((vals[0], vals[1]), vals[2] if len(vals) == 3 else cal_cfg["default_start_radius_px"], "manual")
    elif args.capture_start:
        wait_for_operator("Hold the hand still in the START region and press Enter... ")
        xy, spread, n = measure_hand_position(source, detector, cal_cfg["hold_s"])
        calibration.start_region = Region(tuple(map(float, xy)), cal_cfg["default_start_radius_px"], "hand")
        log.info("Start region centre (hand) = (%.1f, %.1f) px, jitter %.1f px from %d detections", *xy, spread, n)
    elif interactive:
        (cx, cy), (ex, ey) = click_points(image, ["centre of the START region", "edge of the START region"])
        calibration.start_region = Region((cx, cy), float(np.hypot(ex - cx, ey - cy)), "click")

    target_ids = args.target_ids or cfg["recording"]["protocol"]["targets"]
    for text in args.target or []:
        tid, region = parse_target(text, cal_cfg["default_target_radius_px"])
        calibration.targets[tid] = region
    if not args.target:
        for tid in target_ids:
            if args.capture_targets:
                wait_for_operator(f"Hold the hand still AT TARGET {tid} and press Enter... ")
                xy, spread, n = measure_hand_position(source, detector, cal_cfg["hold_s"])
                calibration.targets[tid] = Region(tuple(map(float, xy)), cal_cfg["default_target_radius_px"], "hand")
                log.info("Target %s (hand) = (%.1f, %.1f) px, jitter %.1f px", tid, *xy, spread)
            elif interactive:
                (cx, cy), = click_points(image, [f"centre of TARGET {tid}"])
                calibration.targets[tid] = Region((cx, cy), cal_cfg["default_target_radius_px"], "click")
    if source is not None:
        source.close()
    if detector is not None:
        detector.close()

    out = Path(args.out)
    calibration.save(out, overwrite=args.overwrite)
    log.info("Scale: %.5f cm/px (%.2f px/cm) from %d segment(s)", calibration.cm_per_px, 1 / calibration.cm_per_px, len(segments))
    for seg in segments:
        log.info("  segment %.1f px = %.1f cm -> %.5f cm/px", seg.length_px, seg.distance_cm, seg.cm_per_px)
    for warning in calibration.check():
        log.warning(warning)
    log.info("Calibration saved to %s", out)
    return 0


def _add_calibrate_args(p: argparse.ArgumentParser) -> None:
    from datetime import datetime

    default_out = f"data/calibration/calibration_{datetime.now():%Y%m%d_%H%M%S}.json"
    p.add_argument("--source", choices=["picamera2", "opencv", "video", "synthetic"], help="camera (default from config)")
    p.add_argument("--video", help="video file when --source video")
    p.add_argument("--image", help="calibrate on a saved image instead of the camera")
    p.add_argument("--points", action="append", metavar="X1,Y1,X2,Y2", help="pixel coordinates of two tape marks (repeat for more segments)")
    p.add_argument("--distance-cm", type=float, action="append", help="real distance between the two marks (one per --points)")
    p.add_argument("--segments", type=int, default=2, help="number of tape segments to click (default 2: horizontal + vertical)")
    p.add_argument("--start", metavar="X,Y[,R]", help="start region centre (and radius) in pixels")
    p.add_argument("--target", action="append", metavar="ID=X,Y[,R]", help="a target location in pixels (repeat)")
    p.add_argument("--target-ids", nargs="*", help="target names to click/capture (default: recording.protocol.targets)")
    p.add_argument("--capture-start", action="store_true", help="measure the start region with the tracked hand")
    p.add_argument("--capture-targets", action="store_true", help="measure each target with the tracked hand (recommended)")
    p.add_argument("--detector", help="detector for --capture-* (default from config)")
    p.add_argument("--save-frame", help="also save the calibration image here")
    p.add_argument("--notes", help="free-text notes stored in the calibration file")
    p.add_argument("--out", default=default_out, help="output JSON (default: data/calibration/calibration_<time>.json)")
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing calibration file")


# -----------------------------------------------------------------------------
# record / replay
# -----------------------------------------------------------------------------
@command("record", "record a session of movement trials")
def cmd_record(args, cfg) -> int:
    from .record_session import run_recording_session

    if args.camera_distance_cm is not None:
        cfg["camera"]["pose"]["distance_cm"] = args.camera_distance_cm
    if args.camera_notes:
        cfg["camera"]["pose"]["notes"] = args.camera_notes
    session_dir = run_recording_session(
        cfg,
        participant_id=args.participant,
        source_kind=args.source,
        detector_backend=args.detector,
        calibration_path=args.calibration,
        allow_uncalibrated=args.allow_uncalibrated,
        video=args.video,
        n_trials=args.trials,
        auto=args.auto,
        preview=args.preview,
        save_video=args.save_video or None,
        notes=args.notes or "",
        raw_dir=args.raw_dir,
        session_id=args.session_id,
        seed=args.seed,
    )
    print(session_dir)
    return 0


def _add_record_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--participant", required=True, help="pseudonymous participant ID, e.g. P01 (never a name)")
    p.add_argument("--calibration", help="calibration JSON from `handpredict calibrate`")
    p.add_argument("--allow-uncalibrated", action="store_true", help="record in pixels without a calibration")
    p.add_argument("--source", choices=["picamera2", "opencv", "video", "synthetic"], help="camera (default from config)")
    p.add_argument("--video", help="video file when --source video")
    p.add_argument("--detector", choices=["mediapipe", "hailo", "synthetic"], help="detector (default from config)")
    p.add_argument("--trials", type=int, help="number of trials (default: the full protocol plan)")
    p.add_argument("--auto", action="store_true", help="no prompts or countdown (synthetic data / testing)")
    p.add_argument("--preview", action="store_true", help="show a live preview window (needs a display)")
    p.add_argument("--save-video", action="store_true", help="ALSO store the video frames (privacy: off by default)")
    p.add_argument("--notes", help="free-text session notes")
    p.add_argument("--camera-distance-cm", type=float, help="measured camera distance (stored in session.json)")
    p.add_argument("--camera-notes", help="notes about the camera position / angle")
    p.add_argument("--raw-dir", help="where session folders are created (default: paths.raw_dir)")
    p.add_argument("--session-id", help="override the automatic session id")
    p.add_argument("--seed", type=int, help="seed for the trial order (default: study.seed)")


@command("replay", "re-run a detector on a session's saved videos (creates a new session)")
def cmd_replay(args, cfg) -> int:
    from .record_session import replay_session

    print(replay_session(cfg, args.session, args.detector))
    return 0


def _add_replay_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--session", required=True, help="session folder recorded with --save-video")
    p.add_argument("--detector", required=True, choices=["mediapipe", "hailo", "synthetic"])


# -----------------------------------------------------------------------------
# build-dataset / train
# -----------------------------------------------------------------------------
@command("build-dataset", "merge sessions into a processed dataset (kinematics, segmentation, train/test split)")
def cmd_build_dataset(args, cfg) -> int:
    from .dataset import build_dataset

    out = build_dataset(
        cfg, raw_dir=args.raw_dir, session_dirs=args.sessions, name=args.name, out_root=args.out_root,
        allow_mixed_synthetic=args.allow_mixed_synthetic or None, resplit=args.resplit, plots=not args.no_plots,
    )
    print(out)
    return 0


def _add_build_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--raw-dir", help="folder with session folders (default: paths.raw_dir)")
    p.add_argument("--sessions", nargs="*", help="only these session folders")
    p.add_argument("--name", help="dataset name (default: main, or synthetic for synthetic data)")
    p.add_argument("--out-root", help="where datasets are written (default: paths.processed_dir)")
    p.add_argument("--allow-mixed-synthetic", action="store_true", help="DANGER: allow synthetic and real data together")
    p.add_argument("--resplit", action="store_true", help="make a NEW train/test split (old test results become incomparable)")
    p.add_argument("--no-plots", action="store_true", help="skip the per-trial segmentation plots")


@command("train", "train the ML predictors on the development trials and export them to ONNX")
def cmd_train(args, cfg) -> int:
    from .ml.pipeline import train_all

    out = train_all(cfg, args.dataset, models_root=args.models_root, model_names=args.models)
    print(out)
    return 0


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="data/processed/main", help="processed dataset folder")
    p.add_argument("--models", nargs="*", help="only these models (default: all enabled in ml.models)")
    p.add_argument("--models-root", help="where models are written (default: paths.models_dir)")


@command("evaluate", "compare all models on the test trials: statistics, plots and summary.md")
def cmd_evaluate(args, cfg) -> int:
    from .evaluation import run_evaluation

    out = run_evaluation(
        cfg, args.dataset, models_dir=args.models_dir, out_dir=args.out, split=args.split,
        include_dev=not args.no_dev, allow_mixed_synthetic=args.allow_mixed_synthetic or None,
    )
    print(out / "summary.md")
    return 0


def _add_evaluate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="data/processed/main", help="processed dataset folder")
    p.add_argument("--models-dir", help="trained models (default: paths.models_dir/<dataset name>)")
    p.add_argument("--out", help="results folder (default: paths.results_dir/<dataset name>)")
    p.add_argument("--split", choices=["test", "dev"], default="test",
                   help="test = the real result; dev = development diagnostics (models were trained on these)")
    p.add_argument("--no-dev", action="store_true", help="skip the overfitting diagnostic on development trials")
    p.add_argument("--allow-mixed-synthetic", action="store_true", help="DANGER: evaluate a mixed dataset")


@command("selftest", "run the whole pipeline on SYNTHETIC data in a scratch folder (no camera needed)")
def cmd_selftest(args, cfg) -> int:
    import tempfile

    from .selftest import run_selftest

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="handpredict_selftest_"))
    result = run_selftest(cfg, workdir, n_trials=args.trials, quick=not args.full)
    print(result["summary"].read_text())
    print(f"\nSelf-test files: {workdir}")
    return 0


def _add_selftest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workdir", help="scratch folder (default: a new temporary folder)")
    p.add_argument("--trials", type=int, default=27, help="number of synthetic trials (default 27)")
    p.add_argument("--full", action="store_true", help="use the full training budget from the config (slow)")


ARGUMENT_BUILDERS = {
    "selftest": _add_selftest_args,
    "evaluate": _add_evaluate_args,
    "build-dataset": _add_build_dataset_args,
    "train": _add_train_args,
    "download-models": _add_download_args,
    "check-hardware": _add_check_args,
    "calibrate": _add_calibrate_args,
    "record": _add_record_args,
    "replay": _add_replay_args,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handpredict",
        description="Physics-based vs edge-AI prediction of hand movements (Raspberry Pi 5 + AI HAT+ 2).",
    )
    parser.add_argument("--version", action="version", version=f"handpredict {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (_, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--config", help="YAML file with settings to change (merged over configs/default.yaml)")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="change one setting, e.g. --set camera.fps=90")
        p.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
        p.add_argument("--log-file", help="also write all messages to this file")
        ARGUMENT_BUILDERS.get(name, lambda _p: None)(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose, args.log_file)
    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as exc:
        log.error("Configuration problem: %s", exc)
        return 2
    func, _ = COMMANDS[args.command]
    try:
        return int(func(args, cfg) or 0)
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        return 130
    except Exception as exc:  # show a clear message instead of a long traceback by default
        if args.verbose:
            raise
        log.error("%s: %s", type(exc).__name__, exc)
        log.error("(run again with --verbose for the full error)")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
