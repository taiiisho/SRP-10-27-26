"""Running trained ML models as predictors (no PyTorch needed: ONNX Runtime / scikit-learn).

A ``WindowPredictor`` turns the History into the same window the network was
trained on (``windows.window_at``), runs the model and adds the predicted
displacement to the current position. It implements the same ``predict``
interface as the physics models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..predictor import Prediction, Predictor
from ..trial import History
from ..utils import read_json
from .normalize import Normalizer
from .windows import WindowSet, window_at

log = logging.getLogger(__name__)

RunFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class OnnxNetwork:
    """A network loaded from an .onnx file and run with ONNX Runtime on the CPU."""

    def __init__(self, path: str | Path, threads: int = 1, providers=("CPUExecutionProvider",)) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.path = Path(path)
        self.session = ort.InferenceSession(str(path), options, providers=list(providers))
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.providers = self.session.get_providers()

    def __call__(self, seq_n: np.ndarray, static_n: np.ndarray) -> np.ndarray:
        feed = {self.input_names[0]: seq_n.astype(np.float32), self.input_names[1]: static_n.astype(np.float32)}
        return self.session.run(None, feed)[0]


class WindowPredictor(Predictor):
    """An ML model behind the common Predictor interface."""

    family = "ml"

    def __init__(self, name: str, run: RunFn, norm: Normalizer, hardware: str, info: Mapping[str, Any] | None = None):
        self.name = name
        self.run = run
        self.norm = norm
        self.spec = norm.spec
        self.hardware = hardware
        self.info = dict(info or {})

    def _check_horizons(self, horizons_s: np.ndarray) -> None:
        if len(horizons_s) != self.spec.n_horizons or not np.allclose(horizons_s, self.spec.horizons_s):
            raise ValueError(
                f"{self.name} was trained for horizons {np.array(self.spec.horizons_s) * 1000} ms, "
                f"not {np.asarray(horizons_s) * 1000} ms; retrain after changing horizons_ms"
            )

    def predict(self, history: History, horizons_s: np.ndarray) -> Prediction:
        self._check_horizons(np.asarray(horizons_s))
        built = window_at(history.t, history.xy, history.measured, len(history.t) - 1, self.spec, history.context.target_xy)
        if built is None:
            return Prediction(np.full((self.spec.n_horizons, 2), np.nan), fallback="no_window")
        seq, static = built
        disp = self.predict_displacements(seq[None], static[None])[0]
        return Prediction(history.xy[-1] + disp)

    def predict_displacements(self, seq: np.ndarray, static: np.ndarray) -> np.ndarray:
        """Batch prediction from raw windows: displacements in cm, shape (N, H, 2)."""
        seq_n, static_n = self.norm.inputs(seq, static)
        return self.norm.outputs_to_cm(self.run(seq_n, static_n))

    def predict_windowset(self, ws: WindowSet, batch: int = 8192) -> np.ndarray:
        out = [self.predict_displacements(ws.seq[i : i + batch], ws.static[i : i + batch]) for i in range(0, len(ws), batch)]
        return np.concatenate(out) if out else np.zeros((0, self.spec.n_horizons, 2))

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "hardware": self.hardware, **self.info}


def load_ml_predictors(models_dir: str | Path, names: list[str] | None = None, include_seeds: bool = True) -> dict[str, WindowPredictor]:
    """Load trained models from ``models/<dataset>/`` as predictors.

    Returns e.g. {"gru": ensemble, "gru@seed0": ..., "mlp": ..., "gp": ...}.
    """
    models_dir = Path(models_dir)
    summary_path = models_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found: run `handpredict train` first")
    summary = read_json(summary_path)
    predictors: dict[str, WindowPredictor] = {}
    for name, card in summary["models"].items():
        if names is not None and name not in names:
            continue
        model_dir = models_dir / name
        norm = Normalizer.from_dict(read_json(model_dir / "normalizer.json"))
        if card["kind"] == "gp":
            from .gp import load_gp

            gp = load_gp(model_dir / "gp.joblib")
            predictors[name] = WindowPredictor(name, gp, norm, "cpu (scikit-learn)", {"kind": "gp"})
            continue
        if card.get("ensemble_onnx"):
            predictors[name] = WindowPredictor(
                name, OnnxNetwork(model_dir / card["ensemble_onnx"]), norm, "cpu (onnxruntime)",
                {"kind": card["kind"], "members": len(card["seeds"]), "params": card["params"]},
            )
        if include_seeds or not card.get("ensemble_onnx"):
            for seed, file in zip(card["seeds"], card["seed_onnx"]):
                predictors[f"{name}@seed{seed}"] = WindowPredictor(
                    f"{name}@seed{seed}", OnnxNetwork(model_dir / file), norm, "cpu (onnxruntime)",
                    {"kind": card["kind"], "seed": seed, "params": card["params"]},
                )
    return predictors
