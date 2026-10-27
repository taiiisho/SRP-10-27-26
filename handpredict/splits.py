"""Train/test splits by WHOLE TRIALS (or whole participants), never by frames.

Why not split frames at random? Neighbouring frames of one trial are almost
identical (1/60 s apart). If some frames of a trial were in the training set
and the others in the test set, the model would effectively have seen the
answers, and its test error would look far better than it really is.

* The test set is drawn once, stratified so that every (speed, target)
  combination is represented, and saved in ``splits.json``.
* The split is "sticky": if more trials are recorded later and the dataset is
  rebuilt, trials that already have a split keep it, so the test set can
  never be reshuffled after someone has looked at test results.
* The development (non-test) trials are divided into k cross-validation folds,
  again by whole trials, for model selection.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _largest_remainder(quotas: dict, total: int, rng: np.random.Generator) -> dict:
    """Round per-stratum quotas so they add up to ``total`` (ties broken at random)."""
    floors = {k: int(np.floor(q)) for k, q in quotas.items()}
    remaining = total - sum(floors.values())
    keys = list(quotas)
    order = rng.permutation(len(keys))
    keys = [keys[i] for i in order]
    keys.sort(key=lambda k: quotas[k] - floors[k], reverse=True)
    for k in keys[: max(remaining, 0)]:
        floors[k] += 1
    return floors


def split_units(
    units: pd.DataFrame,
    unit_col: str,
    test_fraction: float,
    seed: int,
    stratify_by: Sequence[str] = (),
    previous: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign each unit (trial or participant) to "test" or "dev".

    ``previous`` maps unit -> split for units that were already assigned;
    they keep their split.
    """
    rng = np.random.default_rng(seed)
    previous = dict(previous or {})
    units = units.drop_duplicates(unit_col).reset_index(drop=True)
    strata_cols = [c for c in stratify_by if c in units.columns]
    if strata_cols:
        units["_stratum"] = units[strata_cols].astype(str).agg("|".join, axis=1)
    else:
        units["_stratum"] = "all"
    total_test = int(round(test_fraction * len(units)))
    quotas = units.groupby("_stratum").size().astype(float) * test_fraction
    targets = _largest_remainder(quotas.to_dict(), total_test, rng)
    assignment: dict[str, str] = {}
    for stratum, group in units.groupby("_stratum", sort=True):
        ids = group[unit_col].astype(str).tolist()
        known = [u for u in ids if u in previous]
        new = [u for u in ids if u not in previous]
        for u in known:
            assignment[u] = previous[u]
        need = max(0, targets[stratum] - sum(previous[u] == "test" for u in known))
        new = [new[i] for i in rng.permutation(len(new))]
        for i, u in enumerate(new):
            assignment[u] = "test" if i < need else "dev"
    return assignment


def assign_folds(units: pd.DataFrame, unit_col: str, k: int, seed: int, stratify_by: Sequence[str] = ()) -> dict[str, int]:
    """Deal development units into k folds, spreading every stratum over the folds."""
    rng = np.random.default_rng(seed + 1)
    units = units.drop_duplicates(unit_col).reset_index(drop=True)
    strata_cols = [c for c in stratify_by if c in units.columns]
    key = units[strata_cols].astype(str).agg("|".join, axis=1) if strata_cols else pd.Series("all", index=units.index)
    folds: dict[str, int] = {}
    position = int(rng.integers(k))
    for stratum in sorted(key.unique()):
        ids = units.loc[key == stratum, unit_col].astype(str).tolist()
        for i in rng.permutation(len(ids)):
            folds[ids[i]] = position % k
            position += 1
    return folds


def make_splits(
    trials: pd.DataFrame,
    split_cfg: Mapping,
    cv_folds: int,
    seed: int,
    previous: Mapping | None = None,
) -> dict:
    """Build the full split description stored in splits.json.

    ``trials`` needs columns trial_id, participant_id and the stratification
    columns. Returns {"assignment": {trial_id: "test"|"dev"}, "folds":
    {trial_id: fold}, ...}.
    """
    group_by = split_cfg["group_by"]
    stratify = list(split_cfg.get("stratify_by", []))
    n_participants = trials["participant_id"].nunique()
    if group_by == "participant" and n_participants < 3:
        log.warning(
            "split.group_by=participant needs at least 3 participants (found %d); splitting by trial instead",
            n_participants,
        )
        group_by = "trial"
    unit_col = "trial_id" if group_by == "trial" else "participant_id"
    prev_units = {}
    if previous and previous.get("group_by") == group_by:
        prev_units = previous.get("unit_assignment", {})
    elif previous:
        log.warning("Previous split used group_by=%s; making a NEW split", previous.get("group_by"))
    unit_strat = stratify if group_by == "trial" else []
    unit_assignment = split_units(trials, unit_col, split_cfg["test_fraction"], seed, unit_strat, prev_units)
    assignment = {str(r.trial_id): unit_assignment[str(getattr(r, unit_col))] for r in trials.itertuples()}

    dev = trials[[assignment[str(t)] == "dev" for t in trials["trial_id"]]]
    fold_unit = unit_col
    if group_by == "participant" and dev["participant_id"].nunique() < cv_folds:
        log.warning("Fewer development participants than CV folds; CV folds are made from trials")
        fold_unit = "trial_id"
    k = min(cv_folds, dev[fold_unit].nunique()) if len(dev) else 0
    fold_of_unit = assign_folds(dev, fold_unit, k, seed, stratify if fold_unit == "trial_id" else []) if k >= 2 else {}
    folds = {str(r.trial_id): fold_of_unit.get(str(getattr(r, fold_unit)), 0) for r in dev.itertuples()}
    return {
        "group_by": group_by,
        "unit": unit_col,
        "stratify_by": stratify,
        "test_fraction": split_cfg["test_fraction"],
        "seed": seed,
        "cv_folds": k,
        "unit_assignment": unit_assignment,
        "assignment": assignment,
        "folds": folds,
        "n_test": sum(v == "test" for v in assignment.values()),
        "n_dev": sum(v == "dev" for v in assignment.values()),
    }


def check_no_overlap(splits: Mapping, trial_ids: Iterable[str] | None = None) -> None:
    """Raise if any trial (or participant, when grouping by participant) appears in both dev and test."""
    assignment = splits["assignment"]
    values = set(assignment.values())
    if not values <= {"test", "dev"}:
        raise ValueError(f"unexpected split labels {values}")
    unit_split = splits.get("unit_assignment", {})
    if trial_ids is not None:
        missing = set(map(str, trial_ids)) - set(assignment)
        if missing:
            raise ValueError(f"{len(missing)} trials have no split assignment")
    if splits.get("group_by") == "participant":
        # every participant's trials must all be on the same side
        if len(set(unit_split.values()) - {"test", "dev"}):
            raise ValueError("bad participant split")
