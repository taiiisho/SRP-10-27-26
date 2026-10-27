"""``handpredict train``: select, train and export every enabled ML model.

Only DEVELOPMENT trials are loaded; the function checks that none of them is
in the test set. Output, per model, in ``models/<dataset>/<model>/``:

    normalizer.json        input/output scales (fitted on development data)
    cv_results.csv         validation error per hyperparameter setting and fold
    cv_summary.csv         mean +/- sd over folds per setting
    cv_histories.csv       learning curves of the cross-validation runs
    final_histories.csv    learning curves of the final (per-seed) models
    seed_<k>.pt / .onnx    one trained copy per random seed
    ensemble.onnx          average of the seed copies (the headline ML model)
    model_card.json        everything needed to reproduce / audit the model

and ``models/<dataset>/training_summary.json`` for all models together.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import config_hash
from ..dataset import check_kinematics_match, load_dataset
from ..utils import git_commit, hardware_description, now_iso, sha256_file, software_versions, write_json
from .windows import WindowSpec

log = logging.getLogger(__name__)


def dataset_fps(dataset) -> float:
    fps = dataset.trials.loc[dataset.trials["included"], "fps"].astype(float)
    if fps.nunique() > 1:
        log.warning("Trials were recorded at different frame rates %s; using the median", sorted(fps.unique()))
    return float(fps.median())


def train_all(
    cfg: Mapping,
    dataset_dir: str | Path,
    models_root: str | Path | None = None,
    model_names: Sequence[str] | None = None,
) -> Path:
    import torch

    from .export import export_onnx, onnx_parity
    from .gp import save_gp, train_gp
    from .models import EnsembleModule, count_parameters
    from .normalize import Normalizer
    from .train import TrainSettings, cross_validate, save_state, train_final, windows_of

    dataset = load_dataset(dataset_dir)
    check_kinematics_match(dataset, cfg)
    dev_ids = dataset.trial_ids("dev")
    test_ids = set(dataset.trial_ids("test"))
    if set(dev_ids) & test_ids:
        raise RuntimeError("development and test trials overlap; the split is broken")
    dev_trials = [dataset.processed_trial(t).data for t in dev_ids]  # test trials are never loaded
    folds = {str(k): int(v) for k, v in dataset.splits["folds"].items()}
    spec = WindowSpec.from_config(cfg, dataset_fps(dataset))
    settings = TrainSettings.from_config(cfg)
    seeds = [int(s) for s in cfg["ml"]["training"]["seeds"]]
    opset = int(cfg["ml"]["export"]["onnx_opset"])
    tolerance = float(cfg["ml"]["export"]["parity_tolerance"])
    out = Path(models_root or cfg["paths"]["models_dir"]) / dataset.name
    out.mkdir(parents=True, exist_ok=True)
    log.info("Training on %d development trials (%d test trials untouched); window = %d samples",
             len(dev_trials), len(test_ids), spec.n_samples)

    plain_dev = windows_of(dev_trials, spec)
    summary = {
        "created": now_iso(),
        "dataset": dataset.name,
        "dataset_path": str(dataset.path),
        "dataset_manifest_sha256": sha256_file(dataset.path / "dataset_manifest.json"),
        "data_origin": dataset.data_origin,
        "dev_trials": dev_ids,
        "n_test_trials_untouched": len(test_ids),
        "window_spec": spec.to_dict(),
        "n_dev_windows": len(plain_dev),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "software": software_versions(),
        "git_commit": git_commit(),
        "hardware": hardware_description(),
        "models": {},
    }
    for name, mcfg in cfg["ml"]["models"].items():
        if not mcfg.get("enabled", True) or (model_names and name not in model_names):
            continue
        start = time.perf_counter()
        model_dir = out / name
        model_dir.mkdir(parents=True, exist_ok=True)
        kind = mcfg["kind"]
        log.info("Model %s (%s)", name, kind)
        if kind == "gp":
            norm = Normalizer.fit(plain_dev, spec, settings.horizon_weighting)
            gp = train_gp(plain_dev, norm, mcfg, seeds[0])
            save_gp(gp, model_dir / "gp.joblib")
            write_json(model_dir / "normalizer.json", norm.to_dict(), overwrite=True)
            card = {"kind": "gp", "params": {k: v for k, v in mcfg.items() if k not in ("enabled", "kind")},
                    "kernel": str(gp.regressor.kernel_), "seeds": [seeds[0]], "seed_onnx": [],
                    "ensemble_onnx": None, "hardware_for_inference": "cpu (scikit-learn)"}
        else:
            selection = cross_validate(kind, mcfg, dev_trials, folds, spec, settings, seeds[0])
            selection.table.to_csv(model_dir / "cv_results.csv", index=False)
            selection.summary.to_csv(model_dir / "cv_summary.csv", index=False)
            pd.concat(selection.histories).to_csv(model_dir / "cv_histories.csv", index=False)
            best_row = selection.summary.iloc[0]
            log.info("  selected %s: CV loss %.4f +/- %.4f, %d epochs", selection.best_params,
                     best_row["mean_val_loss"], best_row["sd_val_loss"], selection.n_epochs)
            nets, norm, histories = train_final(kind, selection.best_params, dev_trials, spec, settings, seeds, selection.n_epochs)
            pd.concat(histories).to_csv(model_dir / "final_histories.csv", index=False)
            write_json(model_dir / "normalizer.json", norm.to_dict(), overwrite=True)
            check = plain_dev.select(np.arange(min(256, len(plain_dev))))
            seq_n, static_n = norm.inputs(check.seq, check.static)
            seed_files, parity = [], {}
            for seed, net in zip(seeds, nets):
                save_state(net, model_dir / f"seed_{seed}.pt")
                path = export_onnx(net, spec, model_dir / f"seed_{seed}.onnx", opset)
                parity[f"seed_{seed}"] = onnx_parity(net, path, seq_n, static_n, tolerance)
                seed_files.append(path.name)
            ensemble_file = None
            if cfg["ml"]["training"]["ensemble"]:
                ensemble = EnsembleModule(nets).eval()
                path = export_onnx(ensemble, spec, model_dir / "ensemble.onnx", opset)
                parity["ensemble"] = onnx_parity(ensemble, path, seq_n, static_n, tolerance)
                ensemble_file = path.name
            log.info("  ONNX parity (max |ONNX - PyTorch|): %s", {k: f"{v:.1e}" for k, v in parity.items()})
            card = {
                "kind": kind,
                "params": selection.best_params,
                "selected_config": selection.best_config,
                "cv": {
                    "folds": int(selection.table["fold"].nunique()),
                    "selected_mean_val_loss": float(best_row["mean_val_loss"]),
                    "selected_sd_val_loss": float(best_row["sd_val_loss"]) if pd.notna(best_row["sd_val_loss"]) else None,
                    "all_configs": json.loads(selection.summary.to_json(orient="records")),
                },
                "n_epochs": selection.n_epochs,
                "seeds": seeds,
                "seed_onnx": seed_files,
                "ensemble_onnx": ensemble_file,
                "onnx_parity_max_abs_diff": parity,
                "onnx_sha256": {f: sha256_file(model_dir / f) for f in seed_files + ([ensemble_file] if ensemble_file else [])},
                "n_parameters": count_parameters(nets[0]),
                "hardware_for_inference": "cpu (onnxruntime)",
                "torch_version": torch.__version__,
            }
        card.update({"training_seconds": round(time.perf_counter() - start, 1), "n_dev_trials": len(dev_trials),
                     "n_dev_windows": len(plain_dev), "window_spec": spec.to_dict()})
        write_json(model_dir / "model_card.json", card, overwrite=True)
        summary["models"][name] = card
    if not summary["models"]:
        raise ValueError("no ML model is enabled in ml.models")
    write_json(out / "training_summary.json", summary, overwrite=True)
    log.info("Models written to %s", out)
    return out
