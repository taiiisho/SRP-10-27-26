"""Training the ML predictors (development trials only).

Procedure for each network type (GRU, MLP):

1. Model selection by grouped k-fold cross-validation on the DEVELOPMENT
   trials: for every hyperparameter setting in the config's small grid, train
   on k-1 folds and measure the error on the held-out fold (whole trials),
   with early stopping. The setting with the lowest mean fold error wins.
   The fold-to-fold spread is reported as mean +/- sd.
2. Final training: the chosen setting is trained on ALL development trials,
   once per random seed, for the median number of epochs that was best in
   cross-validation (no test data is used to decide when to stop).
3. Export every seed model, and their average ("ensemble", the headline ML
   model), to ONNX, and check that ONNX Runtime reproduces PyTorch's outputs.

The test trials are never loaded here.

Loss: the Euclidean distance (cm) between predicted and measured
displacement, the same quantity the evaluation reports, made smooth near
zero so its gradient exists: sqrt(d^2 + eps^2) - eps. Each horizon is
weighted by 1 / (zero-motion error at that horizon) so that the 50 ms
horizon counts as much as the 500 ms one.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..trial import TrialData
from .models import build_network, count_parameters
from .normalize import Normalizer
from .windows import WindowSet, WindowSpec, build_windows, time_scaled_copy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainSettings:
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    smooth_eps: float
    horizon_weighting: str
    torch_threads: int
    augment: bool
    copies: int
    position_noise_cm: float
    speed_range: tuple[float, float]

    @classmethod
    def from_config(cls, cfg: Mapping) -> "TrainSettings":
        tr, aug = cfg["ml"]["training"], cfg["ml"]["augmentation"]
        return cls(
            max_epochs=int(tr["max_epochs"]), patience=int(tr["patience"]), batch_size=int(tr["batch_size"]),
            learning_rate=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]),
            smooth_eps=float(tr["smooth_epsilon_cm"]), horizon_weighting=str(tr["horizon_weighting"]),
            torch_threads=int(tr["torch_threads"]), augment=bool(aug["enabled"]), copies=int(aug["copies"]),
            position_noise_cm=float(aug["position_noise_cm"]) if aug["enabled"] else 0.0,
            speed_range=tuple(aug["speed_scale_range"]),
        )


def hyperparameter_grid(model_cfg: Mapping) -> list[dict[str, Any]]:
    """Every combination of the listed hyperparameter values (lists = values to try)."""
    kind = model_cfg["kind"]
    if kind == "recurrent":
        fixed = {"cell": model_cfg.get("cell", "gru"), "num_layers": model_cfg.get("num_layers", 1),
                 "head_hidden": model_cfg.get("head_hidden", 64)}
        names, values = ["hidden_size", "dropout"], [_as_list(model_cfg["hidden_size"]), _as_list(model_cfg["dropout"])]
    elif kind == "mlp":
        sizes = model_cfg["hidden_sizes"]
        sizes = [sizes] if sizes and not isinstance(sizes[0], (list, tuple)) else sizes
        fixed = {}
        names, values = ["hidden_sizes", "dropout"], [list(sizes), _as_list(model_cfg["dropout"])]
    else:
        raise ValueError(f"no grid for model kind {kind!r}")
    return [{**fixed, **dict(zip(names, combo))} for combo in itertools.product(*values)]


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def augmented_copies(trials: Sequence[TrialData], settings: TrainSettings, rng: np.random.Generator) -> list[TrialData]:
    """Time-stretched copies of the TRAINING trials (mild speed changes)."""
    if not settings.augment or settings.copies <= 0:
        return []
    lo, hi = settings.speed_range
    return [time_scaled_copy(t, float(rng.uniform(lo, hi)), rng) for t in trials for _ in range(settings.copies)]


def windows_of(trials: Sequence[TrialData], spec: WindowSpec) -> WindowSet:
    return WindowSet.concat([build_windows(t, spec) for t in trials], spec)


# -----------------------------------------------------------------------------
# Loss and fitting
# -----------------------------------------------------------------------------
def smooth_euclidean_loss(pred_cm, target_cm, mask, weights, eps: float):
    """Mean over valid (example, horizon) pairs of sqrt(|error|^2 + eps^2) - eps, weighted per horizon."""
    dist = torch.sqrt(((pred_cm - target_cm) ** 2).sum(dim=-1) + eps**2) - eps  # (N, H)
    w = weights[None, :] * mask
    return (dist * w).sum() / w.sum().clamp_min(1e-9)


@dataclass
class FitResult:
    model: torch.nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_loss: float
    seconds: float


class _Tensors:
    def __init__(self, ws: WindowSet, norm: Normalizer) -> None:
        seq, static = norm.inputs(ws.seq, ws.static)
        self.seq = torch.from_numpy(seq)
        self.static = torch.from_numpy(static)
        self.target = torch.from_numpy(ws.target.astype(np.float32))
        self.mask = torch.from_numpy(ws.mask.astype(np.float32))

    def __len__(self) -> int:
        return len(self.seq)


def _evaluate_loss(model, data: _Tensors, out_scale, weights, eps: float, batch: int = 4096) -> float:
    model.eval()
    total, weight = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(data), batch):
            sl = slice(i, i + batch)
            pred = model(data.seq[sl], data.static[sl]) * out_scale[None, :, None]
            dist = torch.sqrt(((pred - data.target[sl]) ** 2).sum(-1) + eps**2) - eps
            w = weights[None, :] * data.mask[sl]
            total += float((dist * w).sum())
            weight += float(w.sum())
    return total / max(weight, 1e-9)


def fit_network(
    model: torch.nn.Module,
    train: WindowSet,
    val: WindowSet | None,
    norm: Normalizer,
    settings: TrainSettings,
    seed: int,
    epochs: int | None = None,
) -> FitResult:
    """Train with Adam. With ``val`` and no ``epochs``: early stopping on the validation loss."""
    torch.manual_seed(seed)
    torch.set_num_threads(settings.torch_threads)
    gen = torch.Generator().manual_seed(seed)
    tr = _Tensors(train, norm)
    va = _Tensors(val, norm) if val is not None and len(val) else None
    out_scale = torch.tensor(norm.out_scale, dtype=torch.float32)
    weights = torch.tensor(norm.horizon_weights, dtype=torch.float32)
    noise_rel = settings.position_noise_cm / norm.pos_scale
    noise_abs = torch.tensor(settings.position_noise_cm / norm.abs_std, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    n_epochs = epochs if epochs is not None else settings.max_epochs
    early_stop = epochs is None and va is not None
    best_val, best_epoch, best_state, wait = np.inf, 0, None, 0
    rows = []
    start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(tr), generator=gen)
        for i in range(0, len(tr), settings.batch_size):
            idx = order[i : i + settings.batch_size]
            seq, static = tr.seq[idx].clone(), tr.static[idx].clone()
            if noise_rel > 0:  # augmentation: a little detector-like jitter on the inputs
                seq[:, :-1, :2] += noise_rel * torch.randn(seq[:, :-1, :2].shape, generator=gen)
                if norm.spec.absolute_position:
                    static[:, 0:2] += noise_abs * torch.randn(static[:, 0:2].shape, generator=gen)
            pred = model(seq, static) * out_scale[None, :, None]
            loss = smooth_euclidean_loss(pred, tr.target[idx], tr.mask[idx], weights, settings.smooth_eps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        train_loss = _evaluate_loss(model, tr, out_scale, weights, settings.smooth_eps)
        val_loss = _evaluate_loss(model, va, out_scale, weights, settings.smooth_eps) if va is not None else np.nan
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if early_stop:
            if val_loss < best_val - 1e-6:
                best_val, best_epoch, best_state, wait = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= settings.patience:
                    break
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(rows)
        best_val = rows[-1]["val_loss"]
    model.eval()
    return FitResult(model, pd.DataFrame(rows), best_epoch, float(best_val), time.perf_counter() - start)


def seeded_network(kind: str, spec: WindowSpec, params: Mapping[str, Any], seed: int) -> torch.nn.Module:
    """Build a network whose random starting weights are fixed by ``seed``.

    The initial weights are drawn when the network is created, so the seed
    must be set right before building it; otherwise the result would depend
    on whatever used the random number generator earlier.
    """
    torch.manual_seed(seed)
    return build_network(kind, spec, params)


def predict_network(model: torch.nn.Module, ws: WindowSet, norm: Normalizer) -> np.ndarray:
    """Predicted displacements in cm, shape (N, H, 2)."""
    seq, static = norm.inputs(ws.seq, ws.static)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(seq), torch.from_numpy(static)).numpy()
    return norm.outputs_to_cm(out)


# -----------------------------------------------------------------------------
# Cross-validation and final training
# -----------------------------------------------------------------------------
@dataclass
class SelectionResult:
    best_params: dict[str, Any]
    best_config: int
    n_epochs: int
    table: pd.DataFrame  # one row per (config, fold)
    summary: pd.DataFrame  # one row per config: mean and sd over folds
    histories: list[pd.DataFrame]


def cross_validate(
    kind: str,
    model_cfg: Mapping,
    dev_trials: Sequence[TrialData],
    folds: Mapping[str, int],
    spec: WindowSpec,
    settings: TrainSettings,
    seed: int,
) -> SelectionResult:
    grid = hyperparameter_grid(model_cfg)
    fold_ids = sorted(set(int(folds[t.context.trial_id]) for t in dev_trials))
    if len(fold_ids) < 2:
        raise ValueError("cross-validation needs at least 2 folds of development trials")
    base_windows = {t.context.trial_id: build_windows(t, spec) for t in dev_trials}
    prepared = {}
    for k in fold_ids:  # the same data for every hyperparameter setting -> a fair comparison
        train_trials = [t for t in dev_trials if folds[t.context.trial_id] != k]
        val_ids = [t.context.trial_id for t in dev_trials if folds[t.context.trial_id] == k]
        aug = augmented_copies(train_trials, settings, np.random.default_rng(seed + 1000 + k))
        plain = WindowSet.concat([base_windows[t.context.trial_id] for t in train_trials], spec)
        train_ws = WindowSet.concat([plain, windows_of(aug, spec)], spec)
        val_ws = WindowSet.concat([base_windows[i] for i in val_ids], spec)
        norm = Normalizer.fit(plain, spec, settings.horizon_weighting)
        prepared[k] = (train_ws, val_ws, norm)
    rows, histories = [], []
    for ci, params in enumerate(grid):
        for k in fold_ids:
            train_ws, val_ws, norm = prepared[k]
            net = seeded_network(kind, spec, params, seed)
            fit = fit_network(net, train_ws, val_ws, norm, settings, seed)
            rows.append({
                "config": ci, "params": json.dumps(params), "fold": k, "best_val_loss": fit.best_val_loss,
                "best_epoch": fit.best_epoch, "n_train_windows": len(train_ws), "n_val_windows": len(val_ws),
                "seconds": round(fit.seconds, 2),
            })
            hist = fit.history.assign(config=ci, fold=k)
            histories.append(hist)
            log.info("  CV config %d %s fold %d: val loss %.4f (epoch %d)", ci, params, k, fit.best_val_loss, fit.best_epoch)
    table = pd.DataFrame(rows)
    summary = table.groupby(["config", "params"]).agg(
        mean_val_loss=("best_val_loss", "mean"), sd_val_loss=("best_val_loss", "std"),
        median_best_epoch=("best_epoch", "median"),
    ).reset_index().sort_values("mean_val_loss")
    best = int(summary.iloc[0]["config"])
    n_epochs = max(5, int(round(summary.iloc[0]["median_best_epoch"])))
    return SelectionResult(grid[best], best, n_epochs, table, summary, histories)


def train_final(
    kind: str,
    params: Mapping[str, Any],
    dev_trials: Sequence[TrialData],
    spec: WindowSpec,
    settings: TrainSettings,
    seeds: Sequence[int],
    n_epochs: int,
) -> tuple[list[torch.nn.Module], Normalizer, list[pd.DataFrame]]:
    """Train the chosen setting on all development trials, once per seed."""
    plain = windows_of(dev_trials, spec)
    norm = Normalizer.fit(plain, spec, settings.horizon_weighting)
    models, histories = [], []
    for seed in seeds:
        aug = augmented_copies(dev_trials, settings, np.random.default_rng(seed + 2000))
        train_ws = WindowSet.concat([plain, windows_of(aug, spec)], spec)
        net = seeded_network(kind, spec, params, int(seed))
        fit = fit_network(net, train_ws, None, norm, settings, int(seed), epochs=n_epochs)
        log.info("  final %s seed %d: %d epochs, train loss %.4f, %d parameters",
                 kind, seed, n_epochs, fit.history["train_loss"].iloc[-1], count_parameters(net))
        models.append(fit.model)
        histories.append(fit.history.assign(seed=int(seed)))
    return models, norm, histories


def save_state(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
