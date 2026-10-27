"""The train/test split must be by whole trials (or participants) and must never overlap."""

import numpy as np
import pandas as pd
import pytest

from handpredict.splits import check_no_overlap, make_splits


def _trials(n_reps=4, participants=("P01",), speeds=("slow", "medium", "fast"), targets=("T1", "T2", "T3")):
    rows = []
    for p in participants:
        for r in range(n_reps):
            for s in speeds:
                for t in targets:
                    rows.append({"trial_id": f"{p}/{s}_{t}_{r}", "participant_id": p, "speed_condition": s, "target_id": t})
    return pd.DataFrame(rows)


SPLIT_CFG = {"test_fraction": 0.3, "group_by": "trial", "stratify_by": ["speed_condition", "target_id"]}


def test_trial_split_has_no_overlap_and_covers_every_condition():
    trials = _trials()
    splits = make_splits(trials, SPLIT_CFG, cv_folds=5, seed=1)
    check_no_overlap(splits, trials["trial_id"])
    test = {t for t, s in splits["assignment"].items() if s == "test"}
    dev = {t for t, s in splits["assignment"].items() if s == "dev"}
    assert not test & dev and test | dev == set(trials["trial_id"])
    assert len(test) == round(0.3 * len(trials))
    test_rows = trials[trials["trial_id"].isin(test)]
    assert len(test_rows.groupby(["speed_condition", "target_id"])) == 9  # every combination is tested
    # CV folds only contain development trials, and every development trial has a fold
    assert set(splits["folds"]) == dev
    sizes = np.bincount(list(splits["folds"].values()))
    assert sizes.max() - sizes.min() <= 1


def test_split_is_reproducible_and_seed_dependent():
    trials = _trials()
    a = make_splits(trials, SPLIT_CFG, 5, seed=1)["assignment"]
    b = make_splits(trials, SPLIT_CFG, 5, seed=1)["assignment"]
    c = make_splits(trials, SPLIT_CFG, 5, seed=2)["assignment"]
    assert a == b and a != c


def test_split_is_sticky_when_new_trials_arrive():
    first = _trials(n_reps=4)
    splits = make_splits(first, SPLIT_CFG, 5, seed=1)
    more = pd.concat([first, _trials(n_reps=6).iloc[len(first):]], ignore_index=True)
    again = make_splits(more, SPLIT_CFG, 5, seed=1, previous=splits)
    for trial_id, side in splits["assignment"].items():
        assert again["assignment"][trial_id] == side  # nobody changes sides
    assert again["n_test"] == pytest.approx(0.3 * len(more), abs=1)


def test_participant_split_keeps_each_participant_on_one_side():
    trials = _trials(n_reps=2, participants=("P01", "P02", "P03", "P04", "P05"))
    cfg = {**SPLIT_CFG, "group_by": "participant"}
    splits = make_splits(trials, cfg, 3, seed=4)
    side = trials.assign(split=trials["trial_id"].map(splits["assignment"]))
    assert (side.groupby("participant_id")["split"].nunique() == 1).all()
    assert splits["group_by"] == "participant"


def test_participant_split_falls_back_to_trials_with_too_few_participants():
    trials = _trials(n_reps=4, participants=("P01", "P02"))
    splits = make_splits(trials, {**SPLIT_CFG, "group_by": "participant"}, 5, seed=4)
    assert splits["group_by"] == "trial"
