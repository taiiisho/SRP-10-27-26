"""``handpredict selftest``: run the whole pipeline on SYNTHETIC data in a scratch folder.

record (synthetic camera + detector) -> build-dataset -> train -> evaluate

It checks that every step works on this computer in a few minutes, without
a camera. Everything it writes is labelled synthetic and lives in its own
folder, so it can never mix with real data. Its "verdict" is meaningless for
the research question.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from .config import apply_overrides

log = logging.getLogger(__name__)

# A small training budget so the self-test finishes in a few minutes.
QUICK_OVERRIDES = [
    "ml.training.max_epochs=25",
    "ml.training.patience=6",
    "ml.training.seeds=[0,1]",
    "ml.training.cv_folds=3",
    "ml.models.gru.hidden_size=[32]",
    "ml.models.mlp.hidden_sizes=[[64,64]]",
    "ml.models.mlp.dropout=[0.0]",
    "ml.models.gp.max_train_points=300",
    "ml.models.gp.n_restarts_optimizer=0",
    "evaluation.bootstrap_samples=500",
]


def run_selftest(cfg: Mapping, workdir: str | Path, n_trials: int = 27, quick: bool = True) -> dict:
    from .dataset import build_dataset
    from .evaluation import run_evaluation
    from .ml.pipeline import train_all
    from .record_session import run_recording_session

    workdir = Path(workdir)
    overrides = list(QUICK_OVERRIDES) if quick else []
    overrides += [
        f"paths.raw_dir={workdir / 'raw'}",
        f"paths.processed_dir={workdir / 'processed'}",
        f"paths.models_dir={workdir / 'models'}",
        f"paths.results_dir={workdir / 'results'}",
    ]
    cfg = apply_overrides(cfg, overrides)
    log.info("SELF-TEST on SYNTHETIC data in %s", workdir)
    session = run_recording_session(
        cfg, participant_id=cfg["synthetic"]["participant_id"], source_kind="synthetic",
        detector_backend="synthetic", auto=True, n_trials=n_trials,
    )
    dataset_dir = build_dataset(cfg, plots=False)
    models_dir = train_all(cfg, dataset_dir)
    results_dir = run_evaluation(cfg, dataset_dir, models_dir)
    log.info("SELF-TEST finished. Remember: synthetic results say nothing about real hands.")
    return {"session": session, "dataset": dataset_dir, "models": models_dir, "results": results_dir,
            "summary": results_dir / "summary.md"}
