"""Test helper: MediaPipe's hand-landmark network on the CPU (LiteRT), with the HailoHefRunner interface.

The Hailo HEF ``hand_landmark_lite`` is a compiled copy of this same network,
so running the TFLite version lets the Hailo detector's cropping, decoding and
tracking logic be tested without the AI HAT+ 2. Needs ``pip install
ai-edge-litert`` and the downloaded ``hand_landmarker.task`` (it contains the
landmark network).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np


class LiteRTLandmarkRunner:
    """uint8 RGB 224x224 crop in -> dict of the four raw outputs (like HailoHefRunner)."""

    def __init__(self, task_path: str | Path) -> None:
        from ai_edge_litert.interpreter import Interpreter

        with zipfile.ZipFile(task_path) as bundle:
            model = bundle.read("hand_landmarks_detector.tflite")
        self.interpreter = Interpreter(model_content=model)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        details = self.interpreter.get_output_details()
        self.outputs = [(d["name"], d["index"]) for d in details]
        self.input_shape = (224, 224, 3)
        self.output_shapes = {d["name"]: tuple(int(v) for v in d["shape"][1:]) for d in details}

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        # The HEF normalises inside the network (x / 255); the TFLite model expects it done here.
        self.interpreter.set_tensor(self.input_index, (image.astype(np.float32) / 255.0)[None])
        self.interpreter.invoke()
        return {name: self.interpreter.get_tensor(index).copy() for name, index in self.outputs}

    def close(self) -> None:
        pass


__all__ = ["LiteRTLandmarkRunner"]
