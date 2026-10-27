"""Logging set-up shared by all command-line tools.

We use the ``logging`` module instead of ``print`` so that every message has a
level (INFO, WARNING, ERROR) and can also be saved to a log file next to the
data it describes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

CONSOLE_FORMAT = "%(levelname)-7s %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(verbose: bool = False, log_file: str | Path | None = None) -> None:
    """Send log messages to the console (and optionally to a file)."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        add_log_file(log_file)

    # Third-party libraries are chatty at DEBUG level.
    for noisy in ("matplotlib", "PIL", "urllib3", "absl", "picamera2", "torch", "onnxscript"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_log_file(log_file: str | Path) -> None:
    """Also write all messages (including DEBUG) to ``log_file``."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(FILE_FORMAT))
    logging.getLogger().addHandler(handler)
