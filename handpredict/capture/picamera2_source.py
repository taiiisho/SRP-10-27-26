"""Raspberry Pi camera (Camera Module 3) through Picamera2.

The camera settings are CONTROLLED VARIABLES of the study, so everything that
could change the image is fixed: frame rate, exposure time, gain, white
balance and focus. The values actually used are read back from the camera
and stored in session.json.

Picamera2 notes (checked against the picamera2 source, Oct 2026):

* The stream format "RGB888" is stored in memory as B, G, R, which is exactly
  OpenCV's BGR layout, so frames can be used with OpenCV directly.
* Each frame's metadata contains ``SensorTimestamp`` (nanoseconds), the time
  the sensor captured it. That is the timestamp we record.
* Picamera2 keeps only the most recent finished frame. If our code does not
  collect frames fast enough, older frames are silently dropped; the recorder
  detects this from gaps in the sensor timestamps.

Lines marked ``VERIFY ON HARDWARE`` could not be tested without a Pi camera.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Mapping

from .base import CapturedFrame, FrameSource, SourceUnavailableError, monotonic_ns

log = logging.getLogger(__name__)


class Picamera2Source(FrameSource):
    """Frames from a Raspberry Pi camera with locked exposure, white balance and focus."""

    name = "picamera2"
    is_live = True

    def __init__(self, cfg: Mapping, camera_index: int = 0) -> None:
        cam = cfg["camera"]
        super().__init__(cam["width"], cam["height"], cam["fps"])
        self.cam_cfg = cam
        self.camera_index = camera_index
        self._picam2 = None
        self._controls_module = None
        self.locked: dict[str, Any] = {}
        self.camera_model: str | None = None
        self._index = 0

    # -- start / stop ------------------------------------------------------------
    def open(self) -> None:
        try:
            from libcamera import controls  # type: ignore[import-not-found]
            from picamera2 import Picamera2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SourceUnavailableError(
                "picamera2 is not available. On the Raspberry Pi run "
                "`sudo apt install python3-picamera2` and create the virtual environment with "
                "`python3 -m venv --system-site-packages .venv`."
            ) from exc
        self._controls_module = controls
        try:
            info = Picamera2.global_camera_info()
            if not info:
                raise SourceUnavailableError(
                    "No Raspberry Pi camera detected. Check the ribbon cable and run `rpicam-hello --list-cameras`."
                )
            self.camera_model = info[self.camera_index].get("Model")
            self._picam2 = Picamera2(self.camera_index)
        except SourceUnavailableError:
            raise
        except Exception as exc:  # picamera2 raises various errors when the camera is missing
            raise SourceUnavailableError(f"Could not open the Raspberry Pi camera: {exc}") from exc

        cam = self.cam_cfg
        frame_us = int(round(1e6 / self.nominal_fps))
        startup_controls: dict[str, Any] = {"FrameDurationLimits": (frame_us, frame_us)}
        exposure, wb, focus = cam["exposure"], cam["white_balance"], cam["focus"]
        if exposure["mode"] == "fixed":
            startup_controls.update(
                AeEnable=False, ExposureTime=int(exposure["exposure_time_us"]), AnalogueGain=float(exposure["analogue_gain"])
            )
        if wb["mode"] == "fixed":
            startup_controls.update(AwbEnable=False, ColourGains=tuple(float(g) for g in wb["colour_gains"]))
        if focus["mode"] == "manual" and "AfMode" in self._picam2.camera_controls:
            startup_controls.update(
                AfMode=controls.AfModeEnum.Manual, LensPosition=float(focus["lens_position_dioptres"])
            )

        sensor = {}
        if cam.get("sensor_output_size"):
            sensor["output_size"] = tuple(int(v) for v in cam["sensor_output_size"])  # VERIFY ON HARDWARE
        config = self._picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            sensor=sensor,
            controls=startup_controls,
            buffer_count=int(cam["buffer_count"]),
        )
        self._picam2.configure(config)
        self._picam2.start()
        log.info("Picamera2 started: %s, %dx%d @ %.0f fps", self.camera_model, self.width, self.height, self.nominal_fps)
        self._lock_auto_controls()
        self._read_back_settings()

    def _lock_auto_controls(self) -> None:
        """For 'lock_after_auto' settings: let the automatic algorithm settle, then freeze its result."""
        exposure, wb = self.cam_cfg["exposure"], self.cam_cfg["white_balance"]
        wants_lock = exposure["mode"] == "lock_after_auto" or wb["mode"] == "lock_after_auto"
        if not wants_lock:
            return
        time.sleep(float(self.cam_cfg["settle_s"]))
        md = self._picam2.capture_metadata()
        lock: dict[str, Any] = {}
        if exposure["mode"] == "lock_after_auto":
            lock.update(AeEnable=False, ExposureTime=int(md["ExposureTime"]), AnalogueGain=float(md["AnalogueGain"]))
        if wb["mode"] == "lock_after_auto":
            lock.update(AwbEnable=False, ColourGains=tuple(float(g) for g in md["ColourGains"]))
        self._picam2.set_controls(lock)
        time.sleep(0.2)  # controls take effect a few frames later
        log.info("Locked automatic camera settings: %s", lock)

    def _read_back_settings(self) -> None:
        md = self._picam2.capture_metadata()
        self.locked = {
            key: md.get(key)
            for key in ("ExposureTime", "AnalogueGain", "ColourGains", "LensPosition", "FrameDuration", "ScalerCrop")
            if key in md
        }
        frame_duration = md.get("FrameDuration")
        if frame_duration:
            actual_fps = 1e6 / frame_duration
            if abs(actual_fps - self.nominal_fps) > 0.5:
                log.warning(
                    "Camera runs at %.1f fps instead of the requested %.0f fps. "
                    "Check camera.sensor_output_size and the exposure time.", actual_fps, self.nominal_fps,
                )

    def close(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            finally:
                self._picam2.close()
                self._picam2 = None

    # -- frames ---------------------------------------------------------------------
    def read(self, timeout_s: float = 1.0) -> CapturedFrame | None:
        if self._picam2 is None:
            raise RuntimeError("Picamera2Source.read() called before open()")
        job = self._picam2.capture_request(wait=False)
        try:
            request = self._picam2.wait(job, timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            log.warning("No frame from the camera within %.1f s", timeout_s)
            return None
        host_ns = monotonic_ns()
        try:
            image = request.make_array("main")
            md = request.get_metadata()
        finally:
            request.release()
        sensor_ns = int(md["SensorTimestamp"])
        meta = {
            "exposure_us": md.get("ExposureTime"),
            "analogue_gain": md.get("AnalogueGain"),
            "lens_position": md.get("LensPosition"),
            # Age of the frame when it reached Python. libcamera documents the
            # sensor timestamp as CLOCK_BOOTTIME-based. VERIFY ON HARDWARE.
            "sensor_age_ms": _sensor_age_ms(sensor_ns),
        }
        frame = CapturedFrame(self._index, sensor_ns, host_ns, image, "sensor", meta)
        self._index += 1
        return frame

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "camera_model": self.camera_model,
                "timestamp_source": "sensor (SensorTimestamp)",
                "sensor_output_size": self.cam_cfg.get("sensor_output_size"),
                "requested": {
                    "exposure": self.cam_cfg["exposure"],
                    "white_balance": self.cam_cfg["white_balance"],
                    "focus": self.cam_cfg["focus"],
                },
                "read_back": {k: _jsonable(v) for k, v in self.locked.items()},
            }
        )
        return info


def _sensor_age_ms(sensor_ns: int) -> float | None:
    try:
        age = (time.clock_gettime_ns(time.CLOCK_BOOTTIME) - sensor_ns) / 1e6
    except (AttributeError, OSError):
        return None
    return age if 0.0 <= age < 1000.0 else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)
