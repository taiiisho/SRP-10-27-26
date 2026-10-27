"""USB webcam through OpenCV.

Webcams do not report a sensor timestamp to OpenCV, so the timestamp is
``time.monotonic_ns()`` taken right after ``grab()`` returns, i.e. when the
frame arrived from the USB driver. That includes a small, roughly constant
transfer delay; the jitter of that delay is what matters for velocities.

Webcam exposure / white-balance / focus controls are driver-specific. The
requested values are applied as far as OpenCV allows, and the values the
camera reports back are written to session.json so they can be checked
(``v4l2-ctl --list-ctrls`` lists what a webcam really supports).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .base import CapturedFrame, FrameSource, SourceUnavailableError, monotonic_ns

log = logging.getLogger(__name__)


class OpenCVCameraSource(FrameSource):
    """Frames from a USB webcam (/dev/videoN)."""

    name = "opencv"
    is_live = True

    def __init__(self, cfg: Mapping, device: int | str | None = None) -> None:
        cam = cfg["camera"]
        super().__init__(cam["width"], cam["height"], cam["fps"])
        self.cam_cfg = cam
        self.device = cam["opencv_device"] if device is None else device
        self._cap = None
        self._index = 0
        self.read_back: dict[str, Any] = {}

    def open(self) -> None:
        import cv2

        api = cv2.CAP_V4L2 if self.cam_cfg.get("opencv_api") == "v4l2" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.device, api)
        if not cap.isOpened():
            raise SourceUnavailableError(
                f"Could not open USB camera {self.device!r}. Is it plugged in? (`ls /dev/video*`)"
            )
        fourcc = self.cam_cfg.get("opencv_fourcc")
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.nominal_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        exposure, wb, focus = self.cam_cfg["exposure"], self.cam_cfg["white_balance"], self.cam_cfg["focus"]
        if exposure["mode"] == "fixed":
            # V4L2: auto_exposure 1 = manual; exposure_time_absolute is in 100 us units.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # VERIFY ON HARDWARE (driver-specific)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure["exposure_time_us"] / 100.0)
        if wb["mode"] == "fixed":
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if focus["mode"] == "manual":
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._cap = cap
        self.read_back = {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
            "autofocus": cap.get(cv2.CAP_PROP_AUTOFOCUS),
        }
        if (self.read_back["width"], self.read_back["height"]) != (self.width, self.height):
            log.warning("Webcam delivers %sx%s instead of %dx%d", self.read_back["width"], self.read_back["height"], self.width, self.height)
        if abs(self.read_back["fps"] - self.nominal_fps) > 0.5:
            log.warning("Webcam reports %.1f fps instead of %.0f fps", self.read_back["fps"], self.nominal_fps)
        log.info("USB camera %s opened: %s", self.device, self.read_back)

    def read(self, timeout_s: float = 1.0) -> CapturedFrame | None:
        if self._cap is None:
            raise RuntimeError("OpenCVCameraSource.read() called before open()")
        if not self._cap.grab():
            return None
        arrival_ns = monotonic_ns()  # timestamp = moment the frame arrived
        ok, image = self._cap.retrieve()
        if not ok:
            return None
        frame = CapturedFrame(self._index, arrival_ns, arrival_ns, image, "monotonic", {})
        self._index += 1
        return frame

    def flush(self) -> None:
        """The driver keeps a couple of old frames; their arrival time would look like 'now'."""
        if self._cap is not None:
            for _ in range(2):  # CAP_PROP_BUFFERSIZE is set to 2
                self._cap.grab()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({"device": self.device, "timestamp_source": "monotonic clock after grab()", "read_back": self.read_back})
        return info
