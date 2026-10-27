"""Shared pytest fixtures."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from handpredict.config import load_config


@pytest.fixture(scope="session")
def cfg() -> dict:
    """The default configuration, exactly as the command-line tools load it."""
    return load_config()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def _quiet_logs(caplog):
    caplog.set_level(logging.WARNING)
    yield
