"""Exporting networks to ONNX and checking that ONNX Runtime gives the same answers.

ONNX is a standard file format for neural networks. On the Pi, the ML
predictor runs from the .onnx file with ONNX Runtime, so the model that is
evaluated is exactly the model that is deployed. After every export the same
inputs are run through PyTorch and ONNX Runtime and the outputs compared.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import torch

from .windows import WindowSpec

log = logging.getLogger(__name__)

INPUT_NAMES = ["seq", "static"]
OUTPUT_NAMES = ["displacement"]


class ParityError(RuntimeError):
    """ONNX Runtime and PyTorch disagree by more than the tolerance."""


def export_onnx(model: torch.nn.Module, spec: WindowSpec, path: str | Path, opset: int = 18) -> Path:
    """Write ``model`` (inputs seq (N, W, 3), static (N, S)) to an ONNX file with a variable batch size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    seq = torch.zeros(3, spec.n_samples, spec.seq_features)
    static = torch.zeros(3, spec.n_static)
    batch = torch.export.Dim("batch", min=1, max=1_000_000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(
                model, (seq, static), str(path), input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                dynamic_shapes=({0: batch}, {0: batch}), opset_version=opset, dynamo=True, verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on the torch version
            log.warning("Modern ONNX exporter failed (%s); using the older TorchScript exporter", exc)
            torch.onnx.export(
                model, (seq, static), str(path), input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                dynamic_axes={"seq": {0: "batch"}, "static": {0: "batch"}, "displacement": {0: "batch"}},
                opset_version=opset, dynamo=False,
            )
    return path


def onnx_parity(model: torch.nn.Module, path: str | Path, seq_n: np.ndarray, static_n: np.ndarray, tolerance: float) -> float:
    """Largest |ONNX - PyTorch| difference on the given (normalised) inputs; raises if above tolerance."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {INPUT_NAMES[0]: seq_n.astype(np.float32), INPUT_NAMES[1]: static_n.astype(np.float32)}
    onnx_out = session.run(None, feed)[0]
    model.eval()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(feed["seq"]), torch.from_numpy(feed["static"])).numpy()
    diff = float(np.max(np.abs(onnx_out - torch_out))) if onnx_out.size else 0.0
    if not np.isfinite(diff) or diff > tolerance:
        raise ParityError(f"ONNX output differs from PyTorch by {diff:.2e} (> {tolerance:.1e}) for {path}")
    return diff
