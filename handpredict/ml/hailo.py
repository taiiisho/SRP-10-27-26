"""Stretch goal: running the ML predictor itself on the Hailo-10H.

Why only the MLP: Hailo's compiler (the Dataflow Compiler, DFC) turns ONNX
networks into a HEF file for the accelerator, but it has limited support for
recurrent layers such as GRU/LSTM. The MLP uses only dense layers and ReLU,
which accelerators handle well.

Two steps, on two machines:

1. ``handpredict export-hailo --step onnx`` (any computer with PyTorch):
   writes ``mlp_flat.onnx`` (ONE input vector of fixed size, batch 1) and
   ``calibration.npy`` (real development windows, used by the compiler to
   choose the 8-bit number ranges).
2. ``handpredict export-hailo --step compile`` (an x86-64 Linux computer
   with the Hailo DFC wheel from the Hailo Developer Zone installed; it does
   NOT run on the Raspberry Pi): parse -> optimise (quantise) -> compile ->
   ``mlp.hef``. Copy the HEF to the Pi.

On the Pi, ``HailoNetwork`` runs the HEF; ``handpredict benchmark --ml-hef``
times it and compares its outputs with the ONNX model (8-bit quantisation
changes the numbers slightly). Lines that could not be tested without the DFC
or the accelerator are marked ``VERIFY ON DFC MACHINE`` / ``VERIFY ON HARDWARE``.
Nothing here ever claims the predictor ran on the accelerator unless it did.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np

from ..utils import read_json, write_json

log = logging.getLogger(__name__)


class HailoNetwork:
    """The exported MLP running on the Hailo device: (seq_n, static_n) -> normalised outputs (N, H, 2)."""

    def __init__(self, hef_path: str | Path, n_horizons: int, timeout_ms: int = 1000) -> None:
        from ..landmarks.hailo_detector import HailoHefRunner

        # FLOAT32 in and out: HailoRT quantises the input and de-quantises the
        # output on the CPU. VERIFY ON HARDWARE.
        self.runner = HailoHefRunner(hef_path, timeout_ms=timeout_ms, input_format="FLOAT32")
        self.n_horizons = int(n_horizons)
        self.output_name = self.runner.output_names[0]

    def __call__(self, seq_n: np.ndarray, static_n: np.ndarray) -> np.ndarray:
        flat = np.concatenate([seq_n.reshape(len(seq_n), -1), static_n], axis=1).astype(np.float32)
        out = np.empty((len(flat), self.n_horizons, 2), dtype=np.float32)
        for i, row in enumerate(flat):  # batch size 1, like the real-time use
            result = self.runner(row.reshape(self.runner.input_shape))
            out[i] = np.asarray(result[self.output_name]).reshape(self.n_horizons, 2)
        return out

    def close(self) -> None:
        self.runner.close()


def export_onnx_for_hailo(cfg: Mapping, models_dir: str | Path, model_name: str = "mlp",
                          out_dir: str | Path | None = None, calib_size: int = 1024) -> Path:
    """Step 1: single-input ONNX of the (ensemble) MLP + calibration data."""
    import torch

    from ..dataset import load_dataset
    from .models import FlatMLP, build_network
    from .normalize import Normalizer
    from .train import windows_of

    models_dir = Path(models_dir)
    card = read_json(models_dir / model_name / "model_card.json")
    if card["kind"] != "mlp":
        raise ValueError(f"only the MLP can be exported for the Hailo accelerator (got {card['kind']})")
    norm = Normalizer.from_dict(read_json(models_dir / model_name / "normalizer.json"))
    spec = norm.spec
    members = []
    for seed in card["seeds"]:
        net = build_network("mlp", spec, card["params"])
        net.load_state_dict(torch.load(models_dir / model_name / f"seed_{seed}.pt"))
        members.append(FlatMLP(net).eval())

    class FlatEnsemble(torch.nn.Module):
        def __init__(self, nets):
            super().__init__()
            self.nets = torch.nn.ModuleList(nets)

        def forward(self, flat):
            return torch.stack([n(flat) for n in self.nets], 0).mean(0)

    model = FlatEnsemble(members).eval()
    n_inputs = spec.n_samples * spec.seq_features + spec.n_static
    out = Path(out_dir or models_dir / model_name / "hailo")
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "mlp_flat.onnx"
    # The older TorchScript exporter is used on purpose: it writes the widely
    # supported ONNX opset 13 with a fixed batch size of 1, which is what an
    # accelerator compiler expects. (Its deprecation warning is expected.)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (torch.zeros(1, n_inputs),), str(onnx_path), input_names=["flat"],
                          output_names=["displacement"], opset_version=13, dynamo=False)

    # Calibration data: real (normalised) development windows, never test data.
    summary = read_json(models_dir / "training_summary.json")
    dataset = load_dataset(summary["dataset_path"])
    dev = [dataset.processed_trial(t).data for t in summary["dev_trials"]]
    ws = windows_of(dev, spec)
    rng = np.random.default_rng(int(cfg["study"]["seed"]))
    pick = rng.choice(len(ws), size=min(calib_size, len(ws)), replace=False)
    seq_n, static_n = norm.inputs(ws.seq[pick], ws.static[pick])
    calib = np.concatenate([seq_n.reshape(len(pick), -1), static_n], axis=1).astype(np.float32)
    np.save(out / "calibration.npy", calib)
    write_json(out / "export_info.json", {"model": model_name, "n_inputs": n_inputs, "n_horizons": spec.n_horizons,
                                           "members": len(members), "calibration_windows": int(len(pick)),
                                           "onnx": onnx_path.name}, overwrite=True)
    log.info("Wrote %s and calibration.npy (%d windows). Next: run `--step compile` on an x86 machine with the Hailo DFC.",
             onnx_path, len(pick))
    return out


def compile_hef(export_dir: str | Path, hw_arch: str = "hailo10h") -> Path:
    """Step 2 (x86 Linux + Hailo Dataflow Compiler): ONNX -> quantised HEF."""
    try:
        from hailo_sdk_client import ClientRunner  # type: ignore[import-not-found]  # VERIFY ON DFC MACHINE
    except ImportError as exc:
        raise RuntimeError(
            "The Hailo Dataflow Compiler (hailo_sdk_client) is not installed. It is a separate download from the "
            "Hailo Developer Zone and runs on x86-64 Linux, not on the Raspberry Pi."
        ) from exc
    export_dir = Path(export_dir)
    info = read_json(export_dir / "export_info.json")
    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(  # VERIFY ON DFC MACHINE
        str(export_dir / info["onnx"]), "handpredict_mlp",
        start_node_names=["flat"], end_node_names=["displacement"],
        net_input_shapes={"flat": [1, info["n_inputs"]]},
    )
    calib = np.load(export_dir / "calibration.npy")
    try:  # the parsed network may expect an image-like layout, e.g. (N, 1, 1, F). VERIFY ON DFC MACHINE
        shape = runner.get_hn_model().get_input_layers()[0].input_shape
        calib = calib.reshape([-1, *[int(s) for s in shape[1:]]])
    except Exception:  # noqa: BLE001 - keep the plain (N, F) layout
        pass
    runner.optimize(calib)
    hef = runner.compile()
    hef_path = export_dir / "mlp.hef"
    hef_path.write_bytes(hef)
    log.info("Compiled %s for %s. Copy it to the Pi and run `handpredict benchmark --ml-hef %s`.", hef_path, hw_arch, hef_path)
    return hef_path
