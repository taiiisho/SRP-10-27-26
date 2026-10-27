"""Real-inference tests of the MediaPipe detector (skipped if MediaPipe or its files are missing).

Run ``handpredict download-models mediapipe test-assets`` once to enable them.
The moving-hand test uses a SEMI-SYNTHETIC video: a real hand photo pasted
along a known reaching path, so the detected track can be checked exactly.
"""

from pathlib import Path

import numpy as np
import pytest

from handpredict.capture.base import CapturedFrame
from handpredict.rawdata import load_session, read_trial_table
from handpredict.record_session import replay_session, run_recording_session
from handpredict.synthetic import ReachSpec, reach_position, render_hand_video

pytestmark = pytest.mark.mediapipe

TEST_IMAGE = Path("models/detectors/test_assets/thumb_up.jpg")


@pytest.fixture(scope="module")
def detector_cfg(cfg):
    pytest.importorskip("mediapipe")
    if not Path(cfg["detector"]["mediapipe"]["model_path"]).exists() or not TEST_IMAGE.exists():
        pytest.skip("run `handpredict download-models mediapipe test-assets` to enable MediaPipe tests")
    return cfg


@pytest.fixture(scope="module")
def hand_image():
    import cv2

    return cv2.imread(str(TEST_IMAGE))


def test_mediapipe_finds_the_knuckle_on_a_real_photo(detector_cfg, hand_image):
    from handpredict.landmarks import make_detector

    with make_detector(detector_cfg, "mediapipe") as det:
        frame = CapturedFrame(0, 1_000_000, 0, hand_image, "test")
        result = det.detect(frame)
    assert result.detected
    h, w = hand_image.shape[:2]
    assert 0 < result.x_px < w and 0 < result.y_px < h
    assert result.landmarks_px.shape == (21, 3)
    np.testing.assert_allclose(result.landmarks_px[9, :2], [result.x_px, result.y_px])
    assert result.latency_ms > 0
    assert det.describe()["hardware"] == "cpu"


@pytest.fixture(scope="module")
def moving_hand_video(detector_cfg, hand_image, tmp_path_factory):
    fps = detector_cfg["camera"]["fps"]
    t = np.arange(int(2.0 * fps)) / fps
    spec = ReachSpec(np.array([400.0, 380.0]), np.array([850.0, 330.0]), np.array([850.0, 330.0]), 0.5, 0.9, 0.0, 0.0)
    path = reach_position(spec, t)
    video = tmp_path_factory.mktemp("video") / "semi_synthetic_reach.mp4"
    render_hand_video(hand_image, str(video), path, fps)
    return video, path


def test_mediapipe_tracks_a_moving_hand_in_a_video(detector_cfg, moving_hand_video, tmp_path):
    video, path = moving_hand_video
    session_dir = run_recording_session(
        detector_cfg, participant_id="SEMISYN01", source_kind="video", detector_backend="mediapipe",
        video=str(video), allow_uncalibrated=True, auto=True, n_trials=1, raw_dir=tmp_path / "raw",
        save_video=True,
    )
    session = load_session(session_dir)
    table = read_trial_table(session.trials[0])
    assert table["detected"].mean() > 0.9
    det = table[table["detected"] == 1]
    true_xy = path[det["frame_index"].to_numpy()]
    offset = det[["x_px", "y_px"]].to_numpy() - true_xy
    # The knuckle sits at a fixed place on the pasted photo, so the detected
    # track must follow the true path up to a constant offset.
    assert np.std(offset, axis=0).max() < 3.0  # px
    meta = session.meta
    assert meta["detector"]["backend"] == "mediapipe" and meta["detector"]["model_sha256"]

    # Replay the saved video through the detector again: a NEW session, same track.
    replay_dir = replay_session(detector_cfg, session_dir, "mediapipe")
    assert replay_dir != session_dir and load_session(replay_dir).meta["replay_of"] == session.session_id
    replayed = read_trial_table(load_session(replay_dir).trials[0])
    both = (replayed["detected"] == 1) & (table["detected"] == 1)
    diff = replayed.loc[both, ["x_px", "y_px"]].to_numpy() - table.loc[both, ["x_px", "y_px"]].to_numpy()
    assert np.median(np.linalg.norm(diff, axis=1)) < 3.0  # px (video compression changes pixels slightly)
