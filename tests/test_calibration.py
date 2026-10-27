import numpy as np
import pytest

from handpredict.calibration import (
    Calibration,
    Region,
    ScaleSegment,
    build_calibration,
    measure_hand_position,
    parse_numbers,
    parse_target,
)
from handpredict.capture.synthetic_source import SyntheticSource
from handpredict.landmarks.synthetic_detector import SyntheticDetector
from handpredict.record_session import synthetic_calibration


def test_scale_from_two_points():
    seg = ScaleSegment((100.0, 200.0), (400.0, 600.0), 50.0)  # 500 px apart
    assert seg.length_px == pytest.approx(500.0)
    assert seg.cm_per_px == pytest.approx(0.1)


def test_several_segments_are_averaged_and_their_disagreement_reported():
    cal = Calibration([ScaleSegment((0, 0), (500, 0), 50.0), ScaleSegment((0, 0), (0, 476.19), 50.0)], (1280, 720))
    assert cal.cm_per_px == pytest.approx(0.5 * (0.1 + 50 / 476.19))
    assert cal.scale_spread == pytest.approx(0.0488, abs=1e-3)
    assert not any("disagree" in w for w in cal.check(max_spread=0.05))
    assert any("disagree" in w for w in cal.check(max_spread=0.02))


def test_roundtrip_and_conversion(tmp_path):
    cal = build_calibration((1280, 720), [[0, 0, 400, 0]], [40.0], [300, 350, 55], {"T1": Region((800, 300), 40)}, 60)
    path = cal.save(tmp_path / "cal.json")
    loaded = Calibration.load(path)
    assert loaded.cm_per_px == pytest.approx(0.1)
    assert loaded.start_region.radius_px == 55 and loaded.start_region.contains((310, 360))
    np.testing.assert_allclose(loaded.px_to_cm([[100, 200]]), [[10, 20]])
    np.testing.assert_allclose(loaded.target_px("T1"), [800, 300])
    with pytest.raises(KeyError):
        loaded.target_px("T9")
    with pytest.raises(FileExistsError):
        cal.save(path)  # calibrations are not silently replaced either


def test_parsing_command_line_values():
    assert parse_numbers("1, 2,3,4", 4) == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        parse_numbers("1,2,3", 4)
    tid, region = parse_target("T2=750,160", default_radius=50)
    assert tid == "T2" and region.center_px == (750.0, 160.0) and region.radius_px == 50


def test_synthetic_calibration_matches_config(cfg):
    cal = synthetic_calibration(cfg)
    assert cal.cm_per_px == pytest.approx(cfg["synthetic"]["cm_per_px"])
    assert set(cal.targets) == set(cfg["synthetic"]["targets_px"])


def test_hand_position_measurement_uses_the_median(cfg):
    source = SyntheticSource(cfg, seed=2)
    with source:
        xy, spread, n = measure_hand_position(source, SyntheticDetector(cfg, seed=3), hold_s=1.0)
    np.testing.assert_allclose(xy, cfg["synthetic"]["start_px"], atol=1.0)
    assert n > 50 and spread < 3 * cfg["detector"]["synthetic"]["noise_px"]
