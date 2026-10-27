import numpy as np

from handpredict.capture import count_dropped_frames
from handpredict.capture.synthetic_source import SyntheticSource


def test_count_dropped_frames():
    interval = 1e9 / 60
    assert count_dropped_frames(int(interval), 60) == 0
    assert count_dropped_frames(int(1.2 * interval), 60) == 0  # normal jitter
    assert count_dropped_frames(int(2.0 * interval), 60) == 1
    assert count_dropped_frames(int(4.1 * interval), 60) == 3


def test_synthetic_source_produces_a_reach(cfg):
    src = SyntheticSource(cfg, seed=3)
    with src:
        src.on_trial_start({"target_id": "T2", "speed_condition": "fast", "pre_roll_s": 0.5})
        frames = [src.read() for _ in range(180)]
    t = np.array([f.t_ns for f in frames]) / 1e9
    xy = np.array([f.meta["true_xy_px"] for f in frames])
    assert np.all(np.diff(t) > 0)
    assert all(f.timestamp_source == "synthetic" and f.meta["synthetic"] for f in frames)
    spec = src.current_spec
    assert spec.onset_s > t[0] + 0.5  # movement starts after "GO"
    before = t < spec.onset_s
    np.testing.assert_allclose(xy[before], np.broadcast_to(spec.start_px, (before.sum(), 2)))
    np.testing.assert_allclose(xy[-1], spec.endpoint_px, atol=1e-6)
    assert src.describe()["synthetic"] is True
