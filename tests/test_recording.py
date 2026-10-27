"""Tests for the trial recorder, session files and reading raw data back."""

import os
import stat

import numpy as np
import pandas as pd
import pytest

from handpredict.config import apply_overrides
from handpredict.rawdata import RawDataError, find_sessions, load_session, load_trial_data, read_trial_table
from handpredict.record_session import run_recording_session, synthetic_calibration
from handpredict.recording import TRIAL_COLUMNS, RawDataExistsError, SessionWriter, validate_participant_id
from handpredict.utils import read_json


@pytest.fixture
def synthetic_session(cfg, tmp_path):
    session_dir = run_recording_session(
        cfg, participant_id="SYN01", source_kind="synthetic", detector_backend="synthetic",
        auto=True, n_trials=4, raw_dir=tmp_path / "raw", seed=11,
    )
    return session_dir


def test_session_files_are_complete_labelled_and_read_only(synthetic_session, cfg):
    files = sorted(p.name for p in synthetic_session.iterdir())
    assert "session.json" in files and "trials_index.csv" in files and "session_end.json" in files
    trial_csvs = [f for f in files if f.endswith(".csv") and f.startswith("trial_0")]
    assert len(trial_csvs) == 4 and all("synthetic" in f for f in trial_csvs)  # synthetic in the file NAME
    assert synthetic_session.name.startswith("synthetic_")
    for name in files:
        mode = (synthetic_session / name).stat().st_mode
        assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), f"{name} is still writable"
    meta = read_json(synthetic_session / "session.json")
    assert meta["data_origin"] == "synthetic" and meta["synthetic"] is True
    for key in ("camera", "detector", "landmark", "calibration", "software", "config", "config_hash", "hardware", "trial_plan"):
        assert key in meta
    assert meta["detector"]["backend"] == "synthetic"
    assert meta["landmark"]["index"] == cfg["landmark"]["index"]
    table = pd.read_csv(synthetic_session / trial_csvs[0])
    assert list(table.columns) == TRIAL_COLUMNS
    assert set(table["data_origin"]) == {"synthetic"}  # ... and in the file CONTENTS


def test_raw_rows_are_consistent(synthetic_session):
    session = load_session(synthetic_session)
    for trial in session.trials:
        table = read_trial_table(trial)
        assert np.all(np.diff(table["t_ns"]) > 0)
        assert table["t_s"].iloc[0] == 0.0
        # interpolated rows are never real detections, and they have positions
        assert not np.any((table["interpolated"] == 1) & (table["detected"] == 1))
        filled = table[table["interpolated"] == 1]
        assert filled[["x_px", "y_px"]].notna().all().all()
        missing = table[(table["detected"] == 0) & (table["interpolated"] == 0)]
        assert missing[["x_px", "y_px"]].isna().all().all()
        np.testing.assert_allclose(table["x_cm"], table["x_px"] * session.calibration.cm_per_px, atol=1e-3)


def test_loading_gives_trial_data_in_cm_with_targets(synthetic_session, cfg):
    session = load_session(synthetic_session)
    assert session.synthetic
    trial = session.trials[0]
    data = load_trial_data(session, trial)
    cm = cfg["synthetic"]["cm_per_px"]
    assert data.context.units == "cm"
    np.testing.assert_allclose(data.context.target_xy, np.array(cfg["synthetic"]["targets_px"][trial.target_id]) * cm)
    assert data.context.data_origin == "synthetic"
    assert data.n > 150 and data.measured.mean() > 0.95


def test_raw_data_is_never_overwritten(synthetic_session, cfg):
    from handpredict.capture.synthetic_source import SyntheticSource
    from handpredict.landmarks.synthetic_detector import SyntheticDetector

    with pytest.raises(RawDataExistsError):
        SessionWriter(cfg, "SYN01", SyntheticSource(cfg), SyntheticDetector(cfg), synthetic_calibration(cfg),
                      raw_dir=synthetic_session.parent, session_id=synthetic_session.name)


def test_edited_raw_file_is_detected(synthetic_session):
    session = load_session(synthetic_session)
    path = session.trials[0].csv_path
    os.chmod(path, 0o644)  # someone makes the file writable and "fixes" a value
    text = path.read_text().replace(",1,0,", ",1,1,", 1)
    path.write_text(text)
    with pytest.raises(RawDataError, match="changed since it was recorded"):
        read_trial_table(session.trials[0])


def test_participant_ids_must_be_pseudonyms():
    assert validate_participant_id("P01") == "P01"
    for bad in ("John Smith", "", "a" * 40, "x/../y"):
        with pytest.raises(ValueError):
            validate_participant_id(bad)


def test_find_sessions(synthetic_session):
    assert find_sessions(synthetic_session.parent) == [synthetic_session]


def test_stop_when_settled_ends_trial_early(cfg, tmp_path):
    settle_cfg = apply_overrides(cfg, ["recording.stop_when_settled=true", "recording.protocol.speeds=[fast]"])
    session_dir = run_recording_session(
        settle_cfg, participant_id="SYN01", source_kind="synthetic", detector_backend="synthetic",
        auto=True, n_trials=2, raw_dir=tmp_path / "raw", seed=5,
    )
    session = load_session(session_dir)
    reasons = [t.info["quality"]["stopped_reason"] for t in session.trials]
    durations = [read_trial_table(t)["t_s"].iloc[-1] for t in session.trials]
    assert reasons == ["settled", "settled"]
    assert max(durations) < cfg["recording"]["pre_roll_s"] + cfg["recording"]["duration_s"]
