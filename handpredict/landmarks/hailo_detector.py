"""Hand-landmark detection on the Raspberry Pi AI HAT+ 2 (Hailo-10H accelerator).

What runs where:

* The neural network (Hailo Model Zoo ``hand_landmark_lite``, which is
  MediaPipe's hand-landmark network compiled for the Hailo-10H) runs on the
  accelerator.
* Cropping the frame around the hand and converting the landmarks back to
  image pixels run on the Pi's CPU (cheap).

The Hailo model zoo has no palm detector, so the hand must first be found:
the crop starts at the calibrated START REGION (every trial begins there),
and afterwards follows the hand using the previous frame's landmarks, exactly
like MediaPipe's own tracking (see ``roi.py``). If the hand is lost, the
search starts again at the start region (or, with ``reacquire: mediapipe``,
MediaPipe on the CPU finds the hand region; the landmark itself still comes
from the Hailo model).

HailoRT API (checked against the HailoRT 5.4.0 Python source, Oct 2026):
``VDevice`` -> ``create_infer_model(hef)`` -> ``configure()`` ->
``create_bindings`` -> ``run([bindings], timeout_ms)``. Lines that could not be
run without the hardware are marked ``VERIFY ON HARDWARE``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..capture.base import CapturedFrame
from ..utils import sha256_file
from .base import DetectorUnavailableError, LandmarkDetector, LandmarkResult
from .roi import RotatedSquare, square_from_landmarks

log = logging.getLogger(__name__)

N_LANDMARKS = 21


# -----------------------------------------------------------------------------
# Decoding the network outputs (pure numpy, unit-tested)
# -----------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class HandLandmarkDecoder:
    """Turns the raw outputs of hand_landmark_lite into (21 x 3 landmarks in crop pixels, presence score).

    MediaPipe's network has four outputs: screen landmarks (63 values, in
    pixels of the 224 x 224 input), a hand-presence score (1 value), a
    handedness score (1 value) and "world" landmarks (63 values, in metres).
    With ``output_map`` = {"landmarks": name, "presence": name} the outputs are
    taken by name. Without it they are recognised by size and value range: the
    screen landmarks are the 63-value output with the larger numbers, and the
    presence score is the first 1-value output in the HEF's output order.
    VERIFY ON HARDWARE: run ``handpredict check-hardware --hef <file>`` and set
    ``detector.hailo.output_map`` if the automatic choice is wrong.
    """

    def __init__(self, input_size: int = 224, output_map: Mapping[str, str] | None = None) -> None:
        self.input_size = int(input_size)
        self.output_map = dict(output_map) if output_map else None

    def __call__(self, outputs: Mapping[str, np.ndarray]) -> tuple[np.ndarray, float]:
        if self.output_map:
            landmarks = np.asarray(outputs[self.output_map["landmarks"]], dtype=float).ravel()
            presence = float(np.asarray(outputs[self.output_map["presence"]], dtype=float).ravel()[0])
        else:
            landmarks, presence = self._guess(outputs)
        if landmarks.size != N_LANDMARKS * 3:
            raise ValueError(f"expected {N_LANDMARKS * 3} landmark values, got {landmarks.size}")
        if not 0.0 <= presence <= 1.0:  # a raw logit: convert to a probability
            presence = _sigmoid(presence)
        return landmarks.reshape(N_LANDMARKS, 3), presence

    @staticmethod
    def _guess(outputs: Mapping[str, np.ndarray]) -> tuple[np.ndarray, float]:
        flat = {name: np.asarray(v, dtype=float).ravel() for name, v in outputs.items()}
        big = [n for n, v in flat.items() if v.size == N_LANDMARKS * 3]
        small = [n for n, v in flat.items() if v.size == 1]
        if not big or not small:
            raise ValueError(f"cannot recognise hand-landmark outputs {[(n, v.size) for n, v in flat.items()]}; "
                             "set detector.hailo.output_map")
        screen = max(big, key=lambda n: float(np.max(np.abs(flat[n]))))
        if float(np.max(np.abs(flat[screen]))) < 2.0:
            raise ValueError("both 63-value outputs look like world landmarks (metres); set detector.hailo.output_map")
        return flat[screen], float(flat[small[0]][0])


# -----------------------------------------------------------------------------
# HailoRT (hardware)
# -----------------------------------------------------------------------------
class HailoHefRunner:
    """Runs one HEF on the Hailo device: UINT8 image in, FLOAT32 outputs out (synchronous)."""

    def __init__(self, hef_path: str | Path, timeout_ms: int = 1000, input_format: str = "UINT8") -> None:
        try:
            from hailo_platform import FormatType, VDevice  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DetectorUnavailableError(
                "hailo_platform (PyHailoRT) is not installed. On the Pi: `sudo apt install hailo-h10-all` and use a "
                "virtual environment created with --system-site-packages."
            ) from exc
        hef_path = Path(hef_path)
        if not hef_path.exists():
            raise DetectorUnavailableError(f"HEF file {hef_path} not found (run `handpredict download-models hailo`)")
        self.hef_path = hef_path
        self.timeout_ms = int(timeout_ms)
        try:
            params = VDevice.create_params()  # VERIFY ON HARDWARE
            self.vdevice = VDevice(params)
            self.infer_model = self.vdevice.create_infer_model(str(hef_path))
            self.infer_model.set_batch_size(1)
            self.infer_model.input().set_format_type(getattr(FormatType, input_format))
            self.output_names = [o.name for o in self.infer_model.outputs]
            for name in self.output_names:
                self.infer_model.output(name).set_format_type(FormatType.FLOAT32)  # dequantised on the host
            self.input_shape = tuple(self.infer_model.input().shape)  # e.g. (224, 224, 3)
            self.output_shapes = {name: tuple(self.infer_model.output(name).shape) for name in self.output_names}
            self._context = self.infer_model.configure()
            self.configured = self._context.__enter__()
            self.buffers = {name: np.empty(shape, dtype=np.float32) for name, shape in self.output_shapes.items()}
            self.bindings = self.configured.create_bindings(output_buffers=self.buffers)
        except DetectorUnavailableError:
            raise
        except Exception as exc:  # HailoRT raises its own exception types
            raise DetectorUnavailableError(
                f"Hailo device not usable ({type(exc).__name__}: {exc}). Check `hailortcli fw-control identify`; "
                "if the HEF was compiled for another HailoRT version, download the matching one."
            ) from exc
        log.info("Hailo model %s loaded: input %s, outputs %s", hef_path.name, self.input_shape, self.output_shapes)

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        self.bindings.input().set_buffer(np.ascontiguousarray(image))
        self.configured.run([self.bindings], self.timeout_ms)  # VERIFY ON HARDWARE
        return {name: np.array(self.bindings.output(name).get_buffer(), copy=True) for name in self.output_names}

    def close(self) -> None:
        context = getattr(self, "_context", None)
        if context is not None:
            context.__exit__(None, None, None)
            self._context = None
        vdevice = getattr(self, "vdevice", None)
        if vdevice is not None:
            vdevice.release()
            self.vdevice = None


def describe_hef(hef_path: str | Path) -> str:
    """Human-readable list of a HEF's input and output layers (for setting output_map)."""
    from hailo_platform import HEF  # type: ignore[import-not-found]

    hef = HEF(str(hef_path))
    lines = []
    for info in hef.get_input_vstream_infos():
        lines.append(f"  input  {info.name}: shape {info.shape}, format {info.format}")
    for info in hef.get_output_vstream_infos():
        lines.append(f"  output {info.name}: shape {info.shape}, format {info.format}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# The detector
# -----------------------------------------------------------------------------
class HailoHandLandmarkDetector(LandmarkDetector):
    """Hand landmarks from the Hailo-10H with crop-based tracking."""

    name = "hailo"
    hardware = "hailo-10h (Raspberry Pi AI HAT+ 2)"
    confidence_meaning = "hand-presence score of the landmark network (0-1)"

    def __init__(self, cfg: Mapping, runner=None) -> None:
        super().__init__(cfg)
        self.hcfg = cfg["detector"]["hailo"]
        self.input_size = int(self.hcfg["input_size"])
        self.runner = runner if runner is not None else HailoHefRunner(self.hcfg["hef_path"], self.hcfg["timeout_ms"])
        self.decoder = HandLandmarkDecoder(self.input_size, self.hcfg.get("output_map"))
        self.threshold = float(self.hcfg["presence_threshold"])
        self.reacquire = str(self.hcfg["reacquire"])
        self.search: RotatedSquare | None = None
        self.track: RotatedSquare | None = None
        self.hef_sha256 = sha256_file(self.hcfg["hef_path"]) if Path(self.hcfg["hef_path"]).exists() else None
        self._cpu_finder = None
        if self.reacquire == "mediapipe":
            from .mediapipe_detector import MediaPipeHandDetector

            self._cpu_finder = MediaPipeHandDetector(cfg)
        self.n_reacquired = 0

    def set_search_region(self, center_px, size_px: float | None = None) -> None:
        """Where to look for the hand when it is not being tracked (the start region)."""
        side = float(size_px or self.hcfg["reacquire_size_px"])
        self.search = RotatedSquare((float(center_px[0]), float(center_px[1])), side, 0.0)

    def reset(self) -> None:
        self.track = None
        if self._cpu_finder is not None:
            self._cpu_finder.reset()

    def _search_squares(self, frame: CapturedFrame) -> list[RotatedSquare]:
        """Crops to try while the hand is not being tracked.

        The landmark network only works on a roughly upright hand of roughly
        the right size, and we do not know how the hand is held when we start
        looking, so several rotations and sizes are tried; the most confident
        one wins. (Only while searching; tracking needs one crop per frame.)
        """
        h, w = frame.image.shape[:2]
        if self.reacquire == "mediapipe" and self._cpu_finder is not None:
            found = self._cpu_finder.detect(frame)
            if not found.detected:
                return []
            return [square_from_landmarks(found.landmarks_px, self.hcfg["roi_scale"], self.hcfg["roi_shift_y"])]
        if self.reacquire == "full_frame":
            base = RotatedSquare((w / 2.0, h / 2.0), float(min(w, h)), 0.0)
        elif self.search is not None:
            base = self.search
        else:  # no start region known: search the image centre
            base = RotatedSquare((w / 2.0, h / 2.0), float(self.hcfg["reacquire_size_px"]), 0.0)
        return [
            RotatedSquare(base.center, base.side * float(scale), np.radians(float(angle)))
            for scale in self.hcfg.get("search_scales", [1.0])
            for angle in self.hcfg.get("search_angles_deg", [0.0])
        ]

    def _run(self, rgb: np.ndarray, square: RotatedSquare) -> tuple[np.ndarray, float]:
        return self.decoder(self.runner(square.crop(rgb, self.input_size)))

    def _detect(self, frame: CapturedFrame) -> LandmarkResult:
        import cv2

        if frame.image is None:
            raise ValueError("HailoHandLandmarkDetector needs real images")
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        if self.track is not None:
            square = self.track
            landmarks_crop, presence = self._run(rgb, square)
        else:
            best = None
            for candidate in self._search_squares(frame):
                landmarks_crop, presence = self._run(rgb, candidate)
                if best is None or presence > best[2]:
                    best = (candidate, landmarks_crop, presence)
            if best is None:
                return LandmarkResult(detected=False)
            square, landmarks_crop, presence = best
            if presence >= self.threshold:
                self.n_reacquired += 1
        if presence < self.threshold:
            self.track = None  # lost: search again next frame
            return LandmarkResult(detected=False, confidence=presence)
        points = square.to_image(landmarks_crop[:, :2], self.input_size)
        z = landmarks_crop[:, 2] * square.side / self.input_size
        landmarks_px = np.column_stack([points, z])
        self.track = square_from_landmarks(landmarks_px, self.hcfg["roi_scale"], self.hcfg["roi_shift_y"])
        lm = landmarks_px[self.landmark_index]
        return LandmarkResult(True, float(lm[0]), float(lm[1]), float(lm[2]), presence, landmarks_px)

    def close(self) -> None:
        if self.runner is not None and hasattr(self.runner, "close"):
            self.runner.close()
        if self._cpu_finder is not None:
            self._cpu_finder.close()

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({
            "hef_path": str(self.hcfg["hef_path"]), "hef_sha256": self.hef_sha256,
            "hef_url": self.hcfg.get("hef_url"), "input_size": self.input_size,
            "reacquire": self.reacquire, "roi_scale": self.hcfg["roi_scale"], "roi_shift_y": self.hcfg["roi_shift_y"],
            "output_map": self.hcfg.get("output_map"),
            "input_shape": getattr(self.runner, "input_shape", None),
            "output_shapes": {k: list(v) for k, v in getattr(self.runner, "output_shapes", {}).items()},
        })
        return info
