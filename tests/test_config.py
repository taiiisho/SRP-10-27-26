import pytest

from handpredict.config import ConfigError, apply_overrides, config_hash, deep_merge, get, load_config


def test_default_config_loads_and_has_required_horizons(cfg):
    assert cfg["horizons_ms"] == [50, 100, 150, 200, 300, 400, 500]
    assert set(cfg["required_horizons_ms"]) <= set(cfg["horizons_ms"])
    assert cfg["physics"]["headline"] == "minimum_jerk_euler"
    assert cfg["ml"]["headline"] == "gru"
    assert cfg["landmark"]["index"] == 9


def test_overrides_parse_yaml_values(cfg):
    out = apply_overrides(cfg, ["horizons_ms=[50, 100]", "required_horizons_ms=[50]", "camera.fps=90"])
    assert out["horizons_ms"] == [50, 100]
    assert out["camera"]["fps"] == 90
    assert cfg["camera"]["fps"] == 60  # the original is not modified


def test_unknown_override_key_is_rejected(cfg):
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["camera.fsp=90"])  # typo must not be silently ignored


def test_invalid_values_are_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["horizons_ms=[100, 50]"])  # unsorted
    with pytest.raises(ConfigError):
        load_config(overrides=["kinematics.segmentation.offset_speed_cm_s=15"])  # offset above onset
    with pytest.raises(ConfigError):
        load_config(overrides=["camera.exposure.exposure_time_us=20000"])  # longer than a frame at 60 fps


def test_yaml_numbers_without_exponent_sign_are_numbers():
    from handpredict.config import load_yaml

    data = load_yaml("a: 2.0e6\nb: 1e-4\nc: 60\nd: 0.5\ne: text\nf: 1.0e+3")
    assert data == {"a": 2.0e6, "b": 1e-4, "c": 60, "d": 0.5, "e": "text", "f": 1000.0}
    assert isinstance(data["c"], int) and isinstance(data["a"], float)


def test_every_numeric_looking_config_value_is_a_number(cfg):
    """Guard against config values that look like numbers but were parsed as text."""
    import re

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk(v, f"{path}[{i}]")
        else:
            yield path, node

    for path, value in walk(cfg):
        if isinstance(value, str):
            assert not re.fullmatch(r"[-+]?[0-9.]+([eE][-+]?[0-9]+)?", value), f"{path} = {value!r} is text"


def test_deep_merge_and_get():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"b": 10}})
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3}
    assert get(merged, "a.c") == 2
    assert get(merged, "a.x", default="missing") == "missing"


def test_config_hash_changes_with_content(cfg):
    other = apply_overrides(cfg, ["camera.fps=30", "camera.exposure.exposure_time_us=4000"])
    assert config_hash(cfg) != config_hash(other)
    assert config_hash(cfg) == config_hash(load_config())
