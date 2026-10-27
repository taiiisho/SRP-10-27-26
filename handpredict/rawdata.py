"""Reading raw sessions back (read-only): session.json, trial tables, flags.

Every trial file's SHA-256 checksum was stored when it was recorded; it is
checked again on loading, so any later edit of raw data is detected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import Calibration
from .recording import TRIAL_COLUMNS
from .synthetic import SYNTHETIC_LABEL
from .trial import TrialContext, TrialData
from .utils import read_json, sha256_file

log = logging.getLogger(__name__)


class RawDataError(RuntimeError):
    """Raw data is missing, malformed or has been modified."""


@dataclass
class RawTrial:
    trial_id: str
    stem: str
    session_id: str
    csv_path: Path
    json_path: Path
    info: dict[str, Any]

    @property
    def speed_condition(self) -> str:
        return str(self.info["speed_condition"])

    @property
    def target_id(self) -> str:
        return str(self.info["target_id"])

    @property
    def participant_id(self) -> str:
        return str(self.info["participant_id"])

    @property
    def video_path(self) -> Path | None:
        video = self.info.get("video")
        return None if not video else self.csv_path.parent / video["file"]


@dataclass
class RawSession:
    path: Path
    meta: dict[str, Any]
    trials: list[RawTrial]
    flags: pd.DataFrame

    @property
    def session_id(self) -> str:
        return str(self.meta["session_id"])

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic")) or self.meta.get("data_origin") == SYNTHETIC_LABEL

    @property
    def calibration(self) -> Calibration | None:
        cal = self.meta.get("calibration")
        return None if cal is None else Calibration.from_dict(cal)

    @property
    def fps(self) -> float:
        return float(self.meta["camera"]["nominal_fps"])

    def flagged(self, trial_id: str) -> list[str]:
        if self.flags.empty:
            return []
        return self.flags.loc[self.flags["trial_id"] == trial_id, "flag"].astype(str).tolist()


def find_sessions(raw_dir: str | Path) -> list[Path]:
    """All session folders (folders containing a session.json) under ``raw_dir``."""
    root = Path(raw_dir)
    if not root.exists():
        return []
    return sorted(p.parent for p in root.glob("*/session.json"))


def load_session(path: str | Path) -> RawSession:
    path = Path(path)
    meta_path = path / "session.json"
    if not meta_path.exists():
        raise RawDataError(f"{path} is not a session folder (no session.json)")
    meta = read_json(meta_path)
    trials = []
    for json_path in sorted(path.glob("trial_*.json")):
        info = read_json(json_path)
        csv_path = json_path.with_suffix(".csv")
        if not csv_path.exists():
            raise RawDataError(f"{csv_path} is missing")
        trials.append(RawTrial(info["trial_id"], info["stem"], info["session_id"], csv_path, json_path, info))
    flags_path = path / "trial_flags.csv"
    flags = pd.read_csv(flags_path) if flags_path.exists() else pd.DataFrame(columns=["trial_id", "flag", "reason"])
    return RawSession(path, meta, trials, flags)


def read_trial_table(trial: RawTrial, verify_checksum: bool = True) -> pd.DataFrame:
    """Read a raw trial CSV, checking that it is complete and unchanged since recording."""
    if verify_checksum and trial.info.get("csv_sha256"):
        digest = sha256_file(trial.csv_path)
        if digest != trial.info["csv_sha256"]:
            raise RawDataError(
                f"{trial.csv_path} has changed since it was recorded (checksum mismatch). Raw data must not be edited."
            )
    table = pd.read_csv(trial.csv_path)
    missing = set(TRIAL_COLUMNS) - set(table.columns)
    if missing:
        raise RawDataError(f"{trial.csv_path} lacks columns {sorted(missing)}")
    return table


def trial_data_from_table(session: RawSession, trial: RawTrial, table: pd.DataFrame) -> TrialData:
    """Convert a raw table into a TrialData in centimetres (or pixels if uncalibrated)."""
    cal = session.calibration
    detected = table["detected"].to_numpy(dtype=int) == 1
    interpolated = table["interpolated"].to_numpy(dtype=int) == 1
    if cal is not None:
        xy = table[["x_cm", "y_cm"]].to_numpy(dtype=float)
        units, scale = "cm", cal.cm_per_px
    else:
        xy = table[["x_px", "y_px"]].to_numpy(dtype=float)
        units, scale = "px", 1.0
    start = None
    target = None
    if cal is not None and cal.start_region is not None:
        start = np.asarray(cal.start_region.center_px, float) * scale
    if cal is not None and trial.target_id in cal.targets:
        target = cal.target_px(trial.target_id) * scale
    context = TrialContext(
        trial_id=trial.trial_id,
        session_id=session.session_id,
        participant_id=trial.participant_id,
        speed_condition=trial.speed_condition,
        target_id=trial.target_id,
        start_xy=start,
        target_xy=target,
        units=units,
        fps=session.fps,
        data_origin=str(trial.info.get("data_origin", session.meta.get("data_origin", "real"))),
        cm_per_px=None if cal is None else cal.cm_per_px,
        extra={"quality": trial.info.get("quality", {}), "reach_spec": trial.info.get("synthetic_reach_spec")},
    )
    return TrialData(context, table["t_s"].to_numpy(dtype=float), xy, detected, interpolated)


def load_trial_data(session: RawSession, trial: RawTrial, verify_checksum: bool = True) -> TrialData:
    return trial_data_from_table(session, trial, read_trial_table(trial, verify_checksum))
