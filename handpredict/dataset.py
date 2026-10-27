"""``build-dataset``: raw sessions -> one processed dataset (kinematics, segmentation, split).

What happens, in order:

1. Find the session folders and refuse to mix synthetic with real data.
2. Compare the CONTROLLED VARIABLES of the sessions (resolution, frame rate,
   camera pose, landmark, detector backend and hardware, scale) and warn
   about every difference.
3. Load every trial (checksums verified), drop rejected / low-quality trials
   with a recorded reason, run the causal filter and the segmentation.
4. Assign trials to the test or development set (sticky, see splits.py) and
   to cross-validation folds.
5. Write frames.csv (per frame: time, position, velocity, acceleration),
   trials.csv (per trial: labels, segmentation, split), splits.json,
   dataset_manifest.json and one segmentation plot per trial.

Raw data is only read, never changed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import config_hash
from .kinematics import KinematicState, Segmentation, make_filter
from .rawdata import RawDataError, find_sessions, load_session, load_trial_data
from .splits import check_no_overlap, make_splits
from .synthetic import SYNTHETIC_LABEL
from .trial import ProcessedTrial, TrialContext, TrialData, process_trial
from .utils import git_commit, now_iso, read_json, sha256_file, software_versions, write_json

log = logging.getLogger(__name__)

FRAME_COLUMNS = [
    "trial_id", "frame_index", "t_s", "x", "y", "measured", "interpolated",
    "x_filt", "y_filt", "vx", "vy", "ax", "ay", "speed", "in_movement",
]


class MixedDataError(RuntimeError):
    """Synthetic and real data would be mixed."""


def _kinematics_signature(cfg: Mapping) -> str:
    return config_hash({"kinematics": cfg["kinematics"], "recording_gap": cfg["recording"]["max_gap_frames"]})


# -----------------------------------------------------------------------------
# Controlled variables
# -----------------------------------------------------------------------------
def _controlled_values(meta: Mapping[str, Any]) -> dict[str, Any]:
    cam, det = meta.get("camera", {}), meta.get("detector", {})
    cal = meta.get("calibration") or {}
    return {
        "resolution": f"{cam.get('width')}x{cam.get('height')}",
        "nominal_fps": cam.get("nominal_fps"),
        "camera_source": cam.get("source"),
        "camera_model": cam.get("camera_model"),
        "camera_pose": json.dumps(meta.get("camera_pose", {}), sort_keys=True),
        "landmark_index": meta.get("landmark", {}).get("index"),
        "detector_backend": det.get("backend"),
        "detector_hardware": det.get("hardware"),
        "detector_fallback_from": det.get("fallback_from"),
        "detector_model_sha256": det.get("model_sha256") or det.get("hef_sha256"),
        "cm_per_px": None if not cal else round(float(cal["cm_per_px"]), 6),
    }


def compare_controlled_variables(metas: Iterable[Mapping[str, Any]], scale_tolerance: float = 0.05) -> list[str]:
    """Plain-language warnings for every controlled variable that differs between sessions."""
    metas = list(metas)
    if len(metas) < 2:
        return []
    values = {m["session_id"]: _controlled_values(m) for m in metas}
    warnings = []
    keys = next(iter(values.values())).keys()
    for key in keys:
        if key == "cm_per_px":
            scales = [v[key] for v in values.values() if v[key] is not None]
            if scales and (max(scales) - min(scales)) / np.mean(scales) > scale_tolerance:
                warnings.append(f"calibration scale differs by more than {100 * scale_tolerance:.0f}% between sessions: "
                                + ", ".join(f"{s}={v[key]}" for s, v in values.items()))
            continue
        distinct = {str(v[key]) for v in values.values()}
        if len(distinct) > 1:
            warnings.append(f"{key} differs between sessions: " + ", ".join(f"{s}={v[key]}" for s, v in values.items()))
    return warnings


# -----------------------------------------------------------------------------
# Building
# -----------------------------------------------------------------------------
def resting_jitter(data: TrialData, onset_index: int | None, margin_s: float = 0.15) -> float:
    """Detector jitter of the still hand before the movement (std per axis, averaged)."""
    end_t = data.t[onset_index] - margin_s if onset_index is not None else data.t[-1]
    rest = data.measured & (data.t < end_t)
    if rest.sum() < 10:
        return float("nan")
    xy = data.xy[rest]
    t = data.t[rest]
    # remove a slow linear drift before measuring the jitter
    detrended = xy - np.polyval(np.polyfit(t, xy, 1), t[:, None]) if len(t) > 2 else xy - xy.mean(0)
    return float(np.mean(np.std(detrended, axis=0, ddof=1)))


def build_dataset(
    cfg: Mapping,
    raw_dir: str | Path | None = None,
    session_dirs: Iterable[str | Path] | None = None,
    name: str | None = None,
    out_root: str | Path | None = None,
    allow_mixed_synthetic: bool | None = None,
    resplit: bool = False,
    plots: bool = True,
) -> Path:
    """Build the processed dataset; returns its folder."""
    raw_dir = Path(raw_dir or cfg["paths"]["raw_dir"])
    session_paths = [Path(p) for p in session_dirs] if session_dirs else find_sessions(raw_dir)
    if not session_paths:
        raise RawDataError(f"no session folders found in {raw_dir}")
    sessions = [load_session(p) for p in session_paths]
    origins = {"synthetic" if s.synthetic else "real" for s in sessions}
    allow_mixed = cfg["evaluation"]["allow_mixed_synthetic"] if allow_mixed_synthetic is None else allow_mixed_synthetic
    if len(origins) > 1 and not allow_mixed:
        raise MixedDataError(
            "Refusing to mix synthetic and real sessions in one dataset "
            f"(synthetic: {[s.session_id for s in sessions if s.synthetic]}). "
            "Synthetic data is only for testing."
        )
    data_origin = "mixed" if len(origins) > 1 else origins.pop()
    name = name or ("synthetic" if data_origin == SYNTHETIC_LABEL else "main")
    if data_origin != "real" and SYNTHETIC_LABEL not in name:
        name = f"{SYNTHETIC_LABEL}_{name}"
    out = Path(out_root or cfg["paths"]["processed_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)

    warnings = compare_controlled_variables([s.meta for s in sessions])
    for w in warnings:
        log.warning("Controlled variable check: %s", w)

    filt = make_filter(cfg)
    max_bad = float(cfg["recording"]["max_bad_frame_fraction"])
    trial_rows, frame_tables, processed = [], [], {}
    for session in sessions:
        for raw_trial in session.trials:
            row: dict[str, Any] = {
                "trial_id": raw_trial.trial_id,
                "session_id": session.session_id,
                "participant_id": raw_trial.participant_id,
                "speed_condition": raw_trial.speed_condition,
                "target_id": raw_trial.target_id,
                "data_origin": raw_trial.info.get("data_origin", session.meta.get("data_origin")),
                "bad_fraction": raw_trial.info.get("quality", {}).get("bad_fraction", np.nan),
                "included": True,
                "exclusion_reason": "",
            }
            flags = session.flagged(raw_trial.trial_id)
            data = load_trial_data(session, raw_trial)
            ctx = data.context
            row.update(
                units=ctx.units, cm_per_px=ctx.cm_per_px, fps=ctx.fps, n_frames=data.n,
                start_x=np.nan if ctx.start_xy is None else ctx.start_xy[0],
                start_y=np.nan if ctx.start_xy is None else ctx.start_xy[1],
                target_x=np.nan if ctx.target_xy is None else ctx.target_xy[0],
                target_y=np.nan if ctx.target_xy is None else ctx.target_xy[1],
            )
            reason = ""
            if "operator_rejected" in flags:
                reason = "operator_rejected"
            elif row["bad_fraction"] > max_bad:
                reason = "too_many_bad_frames"
            elif ctx.target_xy is None:
                reason = "target_not_calibrated"
            pt = process_trial(data, filt, cfg)
            seg = pt.segmentation
            if not reason and not seg.found:
                reason = "no_movement_detected"
            row.update(
                onset_index=seg.onset_index, offset_index=seg.offset_index, peak_index=seg.peak_index,
                offset_found=seg.offset_found,
                onset_t=np.nan if seg.onset_index is None else data.t[seg.onset_index],
                offset_t=np.nan if seg.offset_index is None else data.t[seg.offset_index],
                peak_speed=np.nan if seg.peak_index is None else float(pt.speed[seg.peak_index]),
                resting_jitter=resting_jitter(data, seg.onset_index),
            )
            row["movement_duration_s"] = row["offset_t"] - row["onset_t"]
            if reason:
                row["included"] = False
                row["exclusion_reason"] = reason
                log.info("Excluding %s: %s", raw_trial.trial_id, reason)
            trial_rows.append(row)
            frame_tables.append(_frame_table(pt))
            processed[raw_trial.trial_id] = pt

    trials = pd.DataFrame(trial_rows)
    frames = pd.concat(frame_tables, ignore_index=True)
    included = trials[trials["included"]].copy()
    if included.empty:
        raise RawDataError("no usable trials after quality checks")

    split_path = out / "splits.json"
    previous = None
    if split_path.exists() and not resplit:
        previous = read_json(split_path)
        log.info("Keeping the existing split for trials already in %s", split_path)
    elif split_path.exists():
        log.warning("--resplit: making a NEW train/test split. Results on the old test set are no longer comparable.")
    splits = make_splits(included, cfg["split"], cfg["ml"]["training"]["cv_folds"], int(cfg["study"]["seed"]), previous)
    check_no_overlap(splits, included["trial_id"])
    trials["split"] = trials["trial_id"].map(splits["assignment"]).fillna("excluded")
    trials["fold"] = trials["trial_id"].map(splits["folds"]).astype("Int64")

    frames.to_csv(out / "frames.csv", index=False, float_format="%.9g")
    trials.to_csv(out / "trials.csv", index=False, float_format="%.9g")
    write_json(split_path, splits, overwrite=True)

    dev_jitter = trials.loc[trials["split"] == "dev", "resting_jitter"].median()
    configured = cfg["kinematics"]["kalman_ca"]["measurement_std_cm"]
    jitter_note = None
    if np.isfinite(dev_jitter) and not 0.5 <= dev_jitter / configured <= 2.0:
        jitter_note = (
            f"Measured resting jitter of the detector is {dev_jitter:.3f} (development trials), but "
            f"kinematics.kalman_ca.measurement_std_cm is {configured}. Consider setting it to {dev_jitter:.3f} "
            "and rebuilding (decide this before looking at any test results)."
        )
        log.warning(jitter_note)
    manifest = {
        "name": name,
        "created": now_iso(),
        "data_origin": data_origin,
        "sessions": [
            {"session_id": s.session_id, "path": str(s.path), "session_json_sha256": sha256_file(s.path / "session.json"),
             "synthetic": s.synthetic, "participant_id": s.meta.get("participant_id"),
             "controlled_variables": _controlled_values(s.meta)}
            for s in sessions
        ],
        "controlled_variable_warnings": warnings,
        "n_trials_total": int(len(trials)),
        "n_trials_included": int(trials["included"].sum()),
        "exclusions": trials.loc[~trials["included"], "exclusion_reason"].value_counts().to_dict(),
        "n_test": int((trials["split"] == "test").sum()),
        "n_dev": int((trials["split"] == "dev").sum()),
        "n_participants": int(included["participant_id"].nunique()),
        "units": sorted(included["units"].unique().tolist()),
        "resting_jitter_dev_median": None if not np.isfinite(dev_jitter) else float(dev_jitter),
        "jitter_note": jitter_note,
        "kinematics_signature": _kinematics_signature(cfg),
        "config": cfg,
        "config_hash": config_hash(cfg),
        "software": software_versions(),
        "git_commit": git_commit(),
    }
    write_json(out / "dataset_manifest.json", manifest, overwrite=True)
    if plots:
        plot_segmentations(processed, trials, cfg, out / "segmentation_plots")
    log.info(
        "Dataset %s: %d trials (%d included: %d dev, %d test), %d participants -> %s",
        name, len(trials), manifest["n_trials_included"], manifest["n_dev"], manifest["n_test"],
        manifest["n_participants"], out,
    )
    return out


def _frame_table(pt: ProcessedTrial) -> pd.DataFrame:
    d, k, seg = pt.data, pt.kin, pt.segmentation
    in_movement = np.zeros(d.n, dtype=bool)
    if seg.found:
        in_movement[seg.onset_index : seg.offset_index + 1] = True
    return pd.DataFrame(
        {
            "trial_id": d.context.trial_id,
            "frame_index": np.arange(d.n),
            "t_s": d.t,
            "x": d.xy[:, 0],
            "y": d.xy[:, 1],
            "measured": d.measured.astype(int),
            "interpolated": d.interpolated.astype(int),
            "x_filt": k.position[:, 0],
            "y_filt": k.position[:, 1],
            "vx": k.velocity[:, 0],
            "vy": k.velocity[:, 1],
            "ax": k.acceleration[:, 0],
            "ay": k.acceleration[:, 1],
            "speed": k.speed,
            "in_movement": in_movement.astype(int),
        },
        columns=FRAME_COLUMNS,
    )


def plot_segmentations(processed: Mapping[str, ProcessedTrial], trials: pd.DataFrame, cfg: Mapping, out_dir: Path) -> None:
    """One figure per trial: speed with the thresholds, and x / y over time."""
    from .plotting import OKABE_ITO, new_figure, save_figure

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_cfg = cfg["kinematics"]["segmentation"]
    info = trials.set_index("trial_id")
    for trial_id, pt in processed.items():
        row = info.loc[trial_id]
        fig, (ax1, ax2) = new_figure(8.0, 5.5, nrows=2, sharex=True)
        t = pt.data.t
        ax1.plot(t, pt.speed, color=OKABE_ITO["blue"], label="speed (causal filter)")
        ax1.axhline(seg_cfg["onset_speed_cm_s"], color=OKABE_ITO["vermillion"], ls="--", lw=1, label="onset threshold")
        ax1.axhline(seg_cfg["offset_speed_cm_s"], color=OKABE_ITO["orange"], ls=":", lw=1, label="offset threshold")
        seg = pt.segmentation
        for ax in (ax1, ax2):
            if seg.found:
                ax.axvline(t[seg.onset_index], color=OKABE_ITO["green"], lw=1.2)
                ax.axvline(t[seg.offset_index], color=OKABE_ITO["purple"], lw=1.2)
        unit = pt.context.units
        ax1.set_ylabel(f"speed ({unit}/s)")
        ax1.legend(loc="upper right", fontsize=8)
        m = pt.data.measured
        ax2.plot(t[m], pt.data.xy[m, 0], ".", ms=3, color=OKABE_ITO["sky_blue"], label="x measured")
        ax2.plot(t[m], pt.data.xy[m, 1], ".", ms=3, color=OKABE_ITO["yellow"], label="y measured")
        ax2.plot(t, pt.kin.position[:, 0], color=OKABE_ITO["blue"], lw=1, label="x filtered")
        ax2.plot(t, pt.kin.position[:, 1], color=OKABE_ITO["orange"], lw=1, label="y filtered")
        ax2.set_xlabel("time since the start of the recording (s)")
        ax2.set_ylabel(f"position ({unit})")
        ax2.legend(loc="upper right", fontsize=8, ncol=2)
        status = "included" if row["included"] else f"EXCLUDED: {row['exclusion_reason']}"
        ax1.set_title(f"{trial_id}  ({row['speed_condition']}, {row['target_id']}, {status})", fontsize=9)
        save_figure(fig, out_dir / f"{trial_id.replace('/', '__')}.png")


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
@dataclass
class ProcessedDataset:
    path: Path
    manifest: dict[str, Any]
    trials: pd.DataFrame
    frames: pd.DataFrame
    splits: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def data_origin(self) -> str:
        return str(self.manifest["data_origin"])

    def trial_ids(self, split: str | None = None) -> list[str]:
        sel = self.trials[self.trials["included"]]
        if split is not None:
            sel = sel[sel["split"] == split]
        return sel["trial_id"].tolist()

    def processed_trial(self, trial_id: str) -> ProcessedTrial:
        row = self.trials.set_index("trial_id").loc[trial_id]
        f = self._frames_by_trial().get_group(trial_id).sort_values("frame_index")
        def opt(x, y):
            return None if not (np.isfinite(x) and np.isfinite(y)) else np.array([x, y], float)
        ctx = TrialContext(
            trial_id=trial_id,
            session_id=str(row["session_id"]),
            participant_id=str(row["participant_id"]),
            speed_condition=str(row["speed_condition"]),
            target_id=str(row["target_id"]),
            start_xy=opt(row["start_x"], row["start_y"]),
            target_xy=opt(row["target_x"], row["target_y"]),
            units=str(row["units"]),
            fps=float(row["fps"]),
            data_origin=str(row["data_origin"]),
            cm_per_px=None if pd.isna(row["cm_per_px"]) else float(row["cm_per_px"]),
        )
        measured = f["measured"].to_numpy(int) == 1
        data = TrialData(ctx, f["t_s"].to_numpy(float), f[["x", "y"]].to_numpy(float), measured,
                         f["interpolated"].to_numpy(int) == 1)
        kin = KinematicState(
            f[["x_filt", "y_filt"]].to_numpy(float), f[["vx", "vy"]].to_numpy(float),
            f[["ax", "ay"]].to_numpy(float), measured,
        )
        def idx(v):
            return None if pd.isna(v) else int(v)
        seg = Segmentation(idx(row["onset_index"]), idx(row["offset_index"]), idx(row["peak_index"]),
                           bool(row["offset_found"]) if not pd.isna(row["offset_found"]) else False)
        return ProcessedTrial(data, kin, seg)

    def processed_trials(self, split: str | None = None) -> list[ProcessedTrial]:
        return [self.processed_trial(t) for t in self.trial_ids(split)]

    def _frames_by_trial(self):
        if not hasattr(self, "_grouped"):
            self._grouped = self.frames.groupby("trial_id")
        return self._grouped


def load_dataset(path: str | Path) -> ProcessedDataset:
    path = Path(path)
    for required in ("dataset_manifest.json", "trials.csv", "frames.csv", "splits.json"):
        if not (path / required).exists():
            raise FileNotFoundError(f"{path / required} not found: run `handpredict build-dataset` first")
    trials = pd.read_csv(path / "trials.csv")
    trials["included"] = trials["included"].astype(bool)
    return ProcessedDataset(
        path=path,
        manifest=read_json(path / "dataset_manifest.json"),
        trials=trials,
        frames=pd.read_csv(path / "frames.csv"),
        splits=read_json(path / "splits.json"),
    )


def check_kinematics_match(dataset: ProcessedDataset, cfg: Mapping) -> None:
    """The dataset's filter / segmentation must match the current config (else rebuild)."""
    if dataset.manifest.get("kinematics_signature") != _kinematics_signature(cfg):
        raise RuntimeError(
            "The kinematics settings in the config differ from the ones used to build this dataset. "
            "Run `handpredict build-dataset` again so filtering and segmentation match."
        )
