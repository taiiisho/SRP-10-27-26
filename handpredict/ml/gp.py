"""Gaussian process regression baseline (scikit-learn), named in the proposal.

A Gaussian process predicts the answer for a new window as a weighted average
of the answers of similar training windows, where "similar" is measured by a
kernel (here a radial basis function). Its own hyperparameters (length scale,
noise level) are fitted by maximising the marginal likelihood on the training
data. Exact GPs scale as n^3 with the number of training examples, so the
training windows are randomly subsampled (``max_train_points``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .normalize import Normalizer
from .windows import WindowSet


class GPModel:
    """A fitted scikit-learn GP behaving like a network: (seq_n, static_n) -> normalised outputs."""

    def __init__(self, regressor, n_horizons: int) -> None:
        self.regressor = regressor
        self.n_horizons = n_horizons

    def __call__(self, seq_n: np.ndarray, static_n: np.ndarray) -> np.ndarray:
        x = np.concatenate([seq_n.reshape(len(seq_n), -1), static_n], axis=1)
        y = self.regressor.predict(x)
        return np.asarray(y, dtype=np.float32).reshape(len(x), self.n_horizons, 2)


def train_gp(train: WindowSet, norm: Normalizer, gp_cfg: Mapping[str, Any], seed: int) -> GPModel:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    complete = train.mask.all(axis=1)  # a GP needs an answer for every horizon
    seq_n, static_n = norm.inputs(train.seq[complete], train.static[complete])
    x = np.concatenate([seq_n.reshape(len(seq_n), -1), static_n], axis=1)
    y = norm.outputs_from_cm(train.target[complete]).reshape(len(x), -1)
    rng = np.random.default_rng(seed)
    n_max = int(gp_cfg["max_train_points"])
    if len(x) > n_max:
        keep = rng.choice(len(x), n_max, replace=False)
        x, y = x[keep], y[keep]
    kernel = ConstantKernel(1.0) * RBF(length_scale=float(gp_cfg["length_scale"])) + WhiteKernel(
        noise_level=float(gp_cfg["noise_level"])
    )
    regressor = GaussianProcessRegressor(
        kernel=kernel, normalize_y=False, n_restarts_optimizer=int(gp_cfg["n_restarts_optimizer"]), random_state=seed
    )
    regressor.fit(x, y)
    return GPModel(regressor, norm.spec.n_horizons)


def save_gp(model: GPModel, path: Path) -> None:
    import joblib

    joblib.dump({"regressor": model.regressor, "n_horizons": model.n_horizons}, path)


def load_gp(path: Path) -> GPModel:
    import joblib

    blob = joblib.load(path)  # our own file, written by save_gp (never load untrusted pickles)
    return GPModel(blob["regressor"], int(blob["n_horizons"]))
