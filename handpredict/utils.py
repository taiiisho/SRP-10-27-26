"""Small helpers for reproducibility and safe file handling.

Raw data must never be overwritten, so every writer here refuses to replace an
existing file unless the caller explicitly allows it, and raw files are made
read-only after they are written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import stat
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np

log = logging.getLogger(__name__)

# Packages whose versions are recorded in session.json / results metadata.
TRACKED_PACKAGES = (
    "handpredict", "numpy", "scipy", "pandas", "matplotlib", "opencv-contrib-python",
    "opencv-python", "opencv-python-headless", "mediapipe", "onnxruntime", "onnx",
    "torch", "scikit-learn", "picamera2", "hailort", "pyyaml",
)


class RawDataExistsError(FileExistsError):
    """Raised instead of overwriting existing raw data."""


def now_iso() -> str:
    """Current wall-clock time as an ISO string. Used only for labelling files, never for measurements."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 checksum of a file, recorded so later analyses can prove the raw data is unchanged."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_dir: str | Path | None = None) -> str | None:
    """The git commit of the code that produced a file (with '+dirty' if there are local edits)."""
    repo_dir = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def software_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Versions of Python and the installed packages that matter for the results."""
    versions = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def hardware_description() -> dict[str, Any]:
    """Describe the computer the code runs on (model, OS, CPU)."""
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
    }
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        info["board_model"] = model_file.read_bytes().decode(errors="ignore").strip("\x00").strip()
    os_release = Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text(errors="ignore").splitlines():
            if line.startswith("PRETTY_NAME="):
                info["os"] = line.split("=", 1)[1].strip('"')
    return info


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch so random choices repeat exactly."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def write_json(path: str | Path, obj: Any, overwrite: bool = False) -> Path:
    """Write pretty JSON. Refuses to replace an existing file unless ``overwrite`` is True."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise RawDataExistsError(f"{path} already exists; refusing to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def make_read_only(path: str | Path) -> None:
    """Remove write permission so raw files cannot be edited by accident."""
    path = Path(path)
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
