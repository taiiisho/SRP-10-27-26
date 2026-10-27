"""Loading the YAML configuration.

All tunable numbers live in ``configs/default.yaml``. A run can change them in
two ways, applied in this order on top of the defaults:

1. ``--config my_settings.yaml``: a YAML file with only the keys to change.
2. ``--set a.b.c=value``: single values on the command line (parsed as YAML,
   so ``--set horizons_ms=[50,100]`` gives a list).

The merged config is a plain nested ``dict`` so students can read it like any
other Python dictionary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import IO, Any, Iterable, Mapping

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid."""


class _ConfigLoader(yaml.SafeLoader):
    """PyYAML's safe loader, plus numbers like ``2e6`` / ``2.0e6``.

    PyYAML follows the old YAML 1.1 rules, where ``2.0e6`` (no sign after the
    ``e``) is read as the *string* "2.0e6". Anyone editing the config would
    reasonably expect a number, so we also accept the YAML 1.2 spelling.
    """


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
        |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?)$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_yaml(stream: str | IO[str]) -> Any:
    """Parse YAML text with the config loader (safe: no arbitrary Python objects)."""
    return yaml.load(stream, Loader=_ConfigLoader)  # noqa: S506 - _ConfigLoader is a SafeLoader


def deep_merge(base: dict, update: Mapping) -> dict:
    """Return a copy of ``base`` with ``update`` merged in (nested dicts are merged, other values replaced)."""
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse one ``a.b.c=value`` override into (["a", "b", "c"], parsed_value)."""
    if "=" not in text:
        raise ConfigError(f"Override {text!r} must look like section.key=value")
    key, raw_value = text.split("=", 1)
    keys = [k for k in key.strip().split(".") if k]
    if not keys:
        raise ConfigError(f"Override {text!r} has an empty key")
    return keys, load_yaml(raw_value)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply ``--set`` style overrides. Unknown keys are rejected to catch typos."""
    cfg = copy.deepcopy(cfg)
    for text in overrides:
        keys, value = parse_override(text)
        node = cfg
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise ConfigError(f"Unknown config section {'.'.join(keys)!r} in override {text!r}")
            node = node[k]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise ConfigError(f"Unknown config key {'.'.join(keys)!r} in override {text!r}")
        node[keys[-1]] = value
    return cfg


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> dict:
    """Load the default config, merge an optional user file, then apply overrides."""
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        cfg = load_yaml(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        with open(path, encoding="utf-8") as fh:
            user = load_yaml(fh) or {}
        cfg = deep_merge(cfg, user)
    cfg = apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def get(cfg: Mapping, dotted_key: str, default: Any = None) -> Any:
    """Read ``cfg["a"]["b"]["c"]`` as ``get(cfg, "a.b.c")``."""
    node: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, Mapping) or k not in node:
            return default
        node = node[k]
    return node


def validate_config(cfg: Mapping) -> None:
    """Check the values that would silently break the science if they were wrong."""
    horizons = cfg.get("horizons_ms")
    if not horizons or any((not isinstance(h, (int, float))) or h <= 0 for h in horizons):
        raise ConfigError("horizons_ms must be a non-empty list of positive numbers")
    if list(horizons) != sorted(horizons) or len(set(horizons)) != len(horizons):
        raise ConfigError("horizons_ms must be sorted and contain no duplicates")
    missing = set(cfg.get("required_horizons_ms", [])) - set(horizons)
    if missing:
        raise ConfigError(f"required_horizons_ms {sorted(missing)} are not in horizons_ms")
    if cfg["physics"]["headline"] not in cfg["physics"]["models"]:
        raise ConfigError("physics.headline must be one of physics.models")
    if cfg["ml"]["headline"] not in cfg["ml"]["models"]:
        raise ConfigError("ml.headline must be one of ml.models")
    seg = cfg["kinematics"]["segmentation"]
    if seg["offset_speed_cm_s"] > seg["onset_speed_cm_s"]:
        raise ConfigError("segmentation hysteresis needs offset_speed_cm_s <= onset_speed_cm_s")
    if not 0.0 < cfg["split"]["test_fraction"] < 1.0:
        raise ConfigError("split.test_fraction must be between 0 and 1")
    if cfg["kinematics"]["filter"] not in ("kalman_ca", "causal_polyfit"):
        raise ConfigError("kinematics.filter must be kalman_ca or causal_polyfit")
    fps = cfg["camera"]["fps"]
    exposure = cfg["camera"]["exposure"]
    if exposure["mode"] == "fixed" and exposure["exposure_time_us"] >= 1e6 / fps:
        raise ConfigError("camera.exposure.exposure_time_us must be shorter than one frame (1e6 / fps)")


def config_hash(cfg: Mapping) -> str:
    """A short fingerprint of the config, stored with every result for provenance."""
    canonical = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def resolve_path(cfg: Mapping, key: str) -> Path:
    """Resolve a ``paths.*`` entry relative to the current working directory."""
    return Path(cfg["paths"][key]).expanduser()
