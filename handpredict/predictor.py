"""The one interface every prediction model implements (physics and ML alike).

    prediction = model.predict(history, horizons_s)

``history`` holds only samples up to "now" (see ``handpredict.trial.History``)
and ``horizons_s`` is the list of horizons in seconds (0.05, 0.1, ...). The
result holds one predicted (x, y) position per horizon. Because every model
gets exactly the same History objects, the comparison is fair: same past
data, same frames, same horizons, same ground truth.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .trial import History


@dataclass(frozen=True)
class Prediction:
    """Predicted positions, shape (n_horizons, 2), plus bookkeeping."""

    positions: np.ndarray
    # If a model could not use its main method and used a simpler one instead,
    # the reason is stored here (e.g. "too_few_samples"), so fallbacks are
    # counted and reported rather than hidden.
    fallback: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


class Predictor(abc.ABC):
    """Base class: subclasses implement ``predict``."""

    name: str = "predictor"
    family: str = "physics"  # "physics" | "ml" | "baseline"

    @abc.abstractmethod
    def predict(self, history: History, horizons_s: np.ndarray) -> Prediction:
        """Predict the position at now + each horizon, using only ``history``."""

    def describe(self) -> dict[str, Any]:
        """Settings to save with the results."""
        return {"name": self.name, "family": self.family}


def horizons_to_seconds(horizons_ms) -> np.ndarray:
    return np.asarray(horizons_ms, dtype=float) / 1000.0
