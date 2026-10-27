"""``handpredict benchmark``: is everything fast enough for real-time use on the edge?

A prediction for "where will the hand be in 50 ms" is only useful if it is
ready in less than 50 ms after the frame was captured; otherwise the moment it
predicts has already passed. The benchmark therefore reports:

1. the latency of every predictor on its own (median, 95th percentile, max);
2. the whole live pipeline, stage by stage: waiting for the camera, landmark
   detection, filtering, physics prediction, ML prediction, plus frames per
   second, dropped frames, CPU load and temperature, and the end-to-end
   latency from the frame arriving to both predictions being ready;
3. optionally the ML predictor compiled for the Hailo-10H (``--ml-hef``),
   with how much its 8-bit outputs differ from the ONNX model.

Results go to ``results/benchmark/benchmark_<time>.json`` and ``.md``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .calibration import Calibration
from .capture import count_dropped_frames, make_source
from .capture.base import monotonic_ns
from .kinematics import make_filter
from .landmarks import make_detector
from .ml.windows import build_windows
from .online import OnlinePredictor
from .physics import make_physics_predictors
from .synthetic import make_trial_data, sample_reach_spec
from .trial import TrialContext, make_history, process_trial
from .utils import hardware_description, now_iso, software_versions, write_json

log = logging.getLogger(__name__)


def latency_stats(samples_ms) -> dict:
    a = np.asarray(samples_ms, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def system_status() -> dict:
    status: dict = {"loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None}
    temp = Path("/sys/class/thermal/thermal_zone0/temp")
    if temp.exists():
        status["cpu_temp_c"] = int(temp.read_text().strip()) / 1000.0
    if shutil.which("vcgencmd"):  # Raspberry Pi: 0x0 = never throttled
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
        status["throttled"] = out.stdout.strip()
    return status


def _bench_histories(cfg: Mapping, n: int = 60):
    """Realistic mid-movement histories from a synthetic reach (for timing only)."""
    rng = np.random.default_rng(1)
    spec = sample_reach_spec(rng, cfg["synthetic"], "T2", "medium", cfg["recording"]["pre_roll_s"])
    trial = process_trial(make_trial_data(spec, cfg, rng, "bench", "medium", "T2", noise_px=1.5), make_filter(cfg), cfg)
    frames = np.arange(trial.data.n)
    if trial.segmentation.found:
        frames = frames[(frames >= trial.segmentation.onset_index) & (frames <= trial.segmentation.offset_index)]
    frames = frames[np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)]
    return trial, [make_history(trial, int(j)) for j in frames]


def benchmark_predictors(cfg: Mapping, models_dir: str | Path | None, iterations: int,
                         ml_hef: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    horizons = np.asarray(cfg["horizons_ms"], dtype=float) / 1000.0
    trial, histories = _bench_histories(cfg)
    predictors = dict(make_physics_predictors(cfg))
    hardware = {name: "cpu" for name in predictors}
    extra: dict = {}
    if models_dir and Path(models_dir, "training_summary.json").exists():
        from .ml.inference import load_ml_predictors

        for name, p in load_ml_predictors(models_dir, include_seeds=False).items():
            predictors[name] = p
            hardware[name] = p.hardware
        if ml_hef:
            from .ml.hailo import HailoNetwork
            from .ml.inference import WindowPredictor

            mlp = predictors.get("mlp")
            if mlp is None:
                raise ValueError("--ml-hef needs the trained MLP (models/<dataset>/mlp)")
            hailo = WindowPredictor("mlp@hailo", HailoNetwork(ml_hef, mlp.spec.n_horizons), mlp.norm, "hailo-10h")
            predictors["mlp@hailo"] = hailo
            hardware["mlp@hailo"] = "hailo-10h (AI HAT+ 2)"
            ws = build_windows(trial.data, mlp.spec)
            diff = np.abs(hailo.predict_windowset(ws) - mlp.predict_windowset(ws))
            extra["hailo_vs_onnx_max_abs_diff_cm"] = float(np.max(diff)) if diff.size else None
            extra["hailo_vs_onnx_mean_abs_diff_cm"] = float(np.mean(diff)) if diff.size else None
    rows = []
    for name, model in predictors.items():
        for h in histories[:10]:  # warm-up (first calls are slower)
            model.predict(h, horizons)
        times = []
        for i in range(iterations):
            h = histories[i % len(histories)]
            start = time.perf_counter()
            model.predict(h, horizons)
            times.append((time.perf_counter() - start) * 1000.0)
        stats = latency_stats(times)
        rows.append({"model": name, "runs_on": hardware[name], **stats,
                     "p95_below_50ms": bool(stats.get("p95_ms", np.inf) < 50.0)})
    return pd.DataFrame(rows), extra


def benchmark_pipeline(cfg: Mapping, source_kind: str, detector_backend: str, duration_s: float,
                       models_dir: str | Path | None, calibration_path: str | None, video: str | None) -> dict:
    """Run the live pipeline for ``duration_s`` seconds and time every stage."""
    from .record_session import synthetic_calibration

    calibration = Calibration.load(calibration_path) if calibration_path else (
        synthetic_calibration(cfg) if source_kind == "synthetic" else None)
    source = make_source(cfg, source_kind, video_path=video, render=detector_backend != "synthetic")
    detector = make_detector(cfg, detector_backend)
    scale = calibration.cm_per_px if calibration else 1.0
    context = TrialContext("live", "benchmark", "-", "-", "-",
                           None if not calibration or not calibration.start_region else np.asarray(calibration.start_region.center_px) * scale,
                           None, "cm" if calibration else "px", float(cfg["camera"]["fps"]))
    if calibration and calibration.targets:
        first = sorted(calibration.targets)[0]
        context = TrialContext(**{**context.__dict__, "target_id": first, "target_xy": calibration.target_px(first) * scale})
    if calibration and calibration.start_region:
        detector.set_search_region(calibration.start_region.center_px)
    predictors = {cfg["physics"]["headline"]: make_physics_predictors(cfg)[cfg["physics"]["headline"]]}
    ml_name = None
    if models_dir and calibration and Path(models_dir, "training_summary.json").exists():
        from .ml.inference import load_ml_predictors

        ml = load_ml_predictors(models_dir, include_seeds=False)
        ml_name = cfg["ml"]["headline"] if cfg["ml"]["headline"] in ml else next(iter(ml))
        predictors[ml_name] = ml[ml_name]
    online = OnlinePredictor(cfg, context, predictors)
    horizons = np.asarray(cfg["horizons_ms"], dtype=float) / 1000.0
    stages = {k: [] for k in ("capture_wait", "detection", "filter", "physics_predict", "ml_predict", "end_to_end")}
    dropped, frames, detected = 0, 0, 0
    prev_t, first_t = None, None
    temps, cpu = [], []
    wall0, cpu0 = time.perf_counter(), time.process_time()
    last_status = wall0
    with source, detector:
        if hasattr(source, "on_trial_start"):
            source.on_trial_start({"target_id": context.target_id if context.target_id != "-" else "T1",
                                   "speed_condition": "medium", "pre_roll_s": 0.5})
        while time.perf_counter() - wall0 < duration_s:
            t0 = time.perf_counter()
            frame = source.read(timeout_s=2.0)
            stages["capture_wait"].append((time.perf_counter() - t0) * 1000.0)
            if frame is None:
                break
            frames += 1
            first_t = frame.t_ns if first_t is None else first_t
            if prev_t is not None:
                dropped += count_dropped_frames(frame.t_ns - prev_t, source.nominal_fps)
            prev_t = frame.t_ns
            result = detector.detect(frame)
            stages["detection"].append(result.latency_ms)
            detected += int(result.detected)
            t1 = time.perf_counter()
            xy = np.array([result.x_px, result.y_px]) * scale if result.detected else None
            online.update((frame.t_ns - first_t) / 1e9, xy)
            stages["filter"].append((time.perf_counter() - t1) * 1000.0)
            for name, key in ((cfg["physics"]["headline"], "physics_predict"), (ml_name, "ml_predict")):
                if name is None:
                    continue
                t2 = time.perf_counter()
                online.predict(horizons, [name])
                stages[key].append((time.perf_counter() - t2) * 1000.0)
            stages["end_to_end"].append((monotonic_ns() - frame.host_ns) / 1e6)
            if time.perf_counter() - last_status >= 1.0:
                status = system_status()
                temps.append(status.get("cpu_temp_c"))
                last_status = time.perf_counter()
            if not source.is_live and source.name == "synthetic":
                # pace synthetic frames like a real camera so the numbers are comparable
                time.sleep(max(0.0, 1.0 / source.nominal_fps - (time.perf_counter() - t0)))
    wall = time.perf_counter() - wall0
    cpu_fraction = (time.process_time() - cpu0) / wall if wall > 0 else float("nan")
    cpu.append(cpu_fraction)
    return {
        "source": source.describe(), "detector": detector.describe(), "ml_model": ml_name,
        "frames": frames, "seconds": wall, "fps": frames / wall if wall > 0 else float("nan"),
        "nominal_fps": source.nominal_fps, "dropped_frames": dropped,
        "dropped_fraction": dropped / max(frames + dropped, 1), "detection_rate": detected / max(frames, 1),
        "stages": {k: latency_stats(v) for k, v in stages.items()},
        "process_cpu_cores_used": cpu_fraction,
        "cpu_temp_c_max": max([t for t in temps if t is not None], default=None),
        "system_status_end": system_status(),
    }


def write_report(out_dir: Path, predictor_table: pd.DataFrame, pipeline: dict | None, extra: dict, cfg: Mapping) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shortest = float(min(cfg["horizons_ms"]))
    lines = ["# Edge benchmark", "", f"{now_iso()} on {hardware_description().get('board_model', hardware_description()['machine'])}", ""]
    lines += ["## Predictor latency (one prediction for all horizons)", "",
              "| model | runs on | p50 (ms) | p95 (ms) | max (ms) | p95 < 50 ms? |", "|---|---|---:|---:|---:|---|"]
    for r in predictor_table.itertuples():
        lines.append(f"| {r.model} | {r.runs_on} | {r.p50_ms:.3f} | {r.p95_ms:.3f} | {r.max_ms:.3f} | {'yes' if r.p95_below_50ms else 'NO'} |")
    if extra.get("hailo_vs_onnx_max_abs_diff_cm") is not None:
        lines += ["", f"Hailo (8-bit) vs ONNX (float) MLP outputs: mean |difference| {extra['hailo_vs_onnx_mean_abs_diff_cm']:.3f} cm, "
                      f"max {extra['hailo_vs_onnx_max_abs_diff_cm']:.3f} cm."]
    if pipeline:
        s = pipeline["stages"]
        temp = pipeline["cpu_temp_c_max"]
        temp_text = f"{temp:.1f} C" if temp is not None else "n/a"
        lines += ["", "## Live pipeline", "",
                  f"- Source `{pipeline['source']['source']}`, detector `{pipeline['detector']['backend']}` on "
                  f"{pipeline['detector']['hardware']}" + (f" (fallback from {pipeline['detector']['fallback_from']})" if pipeline['detector'].get('fallback_from') else ""),
                  f"- {pipeline['fps']:.1f} frames/s processed (camera {pipeline['nominal_fps']:.0f} fps), "
                  f"{100 * pipeline['dropped_fraction']:.1f}% frames dropped, hand detected in {100 * pipeline['detection_rate']:.0f}% of frames",
                  f"- CPU: {pipeline['process_cpu_cores_used']:.2f} cores used by this program; max CPU temperature "
                  f"{temp_text}", "",
                  "| stage | p50 (ms) | p95 (ms) | max (ms) |", "|---|---:|---:|---:|"]
        for name, st in s.items():
            if st.get("n"):
                lines.append(f"| {name} | {st['p50_ms']:.2f} | {st['p95_ms']:.2f} | {st['max_ms']:.2f} |")
        e2e = s["end_to_end"].get("p95_ms")
        if e2e is not None:
            margin = shortest - e2e
            verdict = (f"the {shortest:g} ms prediction is ready {margin:.1f} ms before the moment it predicts"
                       if margin > 0 else f"the {shortest:g} ms prediction arrives {-margin:.1f} ms TOO LATE to be useful")
            lines += ["", f"**Frame arrival -> both predictions ready: p95 = {e2e:.1f} ms, so {verdict}.** "
                          "(The sensor exposure and transfer before the frame arrives add a few more ms.)"]
    text = "\n".join(lines) + "\n"
    (out_dir / f"benchmark_{stamp}.md").write_text(text, encoding="utf-8")
    write_json(out_dir / f"benchmark_{stamp}.json", {
        "created": now_iso(), "predictors": predictor_table.to_dict(orient="records"), "pipeline": pipeline,
        "extra": extra, "hardware": hardware_description(), "software": software_versions(),
    }, overwrite=True)
    return out_dir / f"benchmark_{stamp}.md"
