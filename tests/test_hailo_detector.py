"""Tests of the Hailo detector's CPU-side logic (cropping, decoding, tracking), without the Hailo hardware.

The network itself is replaced by the same MediaPipe landmark network run with
LiteRT (tests/litert_runner.py). What can only be checked on the real AI HAT+ 2
(HailoRT calls, output names/order of the HEF, quantisation) is marked
VERIFY ON HARDWARE in hailo_detector.py.
"""

from pathlib import Path

import numpy as np
import pytest

from handpredict.capture.base import CapturedFrame
from handpredict.landmarks.hailo_detector import HailoHandLandmarkDetector, HandLandmarkDecoder
from handpredict.landmarks.roi import RotatedSquare, hand_rotation, square_from_landmarks

TEST_IMAGE = Path("models/detectors/test_assets/thumb_up.jpg")


# -- crop geometry -----------------------------------------------------------------
def test_crop_and_inverse_mapping_are_consistent(rng):
    sq = RotatedSquare((640.0, 360.0), 300.0, 0.7)
    m_in, m_out = sq.image_to_crop_matrix(224), sq.crop_to_image_matrix(224)
    pts = rng.uniform(0, 224, (50, 2))
    back = (pts @ m_out[:, :2].T + m_out[:, 2]) @ m_in[:, :2].T + m_in[:, 2]
    np.testing.assert_allclose(back, pts, atol=1e-9)
    np.testing.assert_allclose(sq.to_image(np.array([[112.0, 112.0]]), 224), [[640.0, 360.0]])


def test_a_bright_dot_lands_where_the_matrix_says():
    image = np.zeros((720, 1280, 3), np.uint8)
    image[395:406, 695:706] = 255  # dot centred at (700, 400)
    sq = RotatedSquare((640.0, 360.0), 300.0, 0.4)
    crop = sq.crop(image, 224)
    ys, xs = np.nonzero(crop[:, :, 0] > 128)
    found = np.array([xs.mean(), ys.mean()])
    expected = np.array([700.0, 400.0]) @ sq.image_to_crop_matrix(224)[:, :2].T + sq.image_to_crop_matrix(224)[:, 2]
    np.testing.assert_allclose(found, expected, atol=1.0)


def test_rotation_turns_the_hand_upright():
    hand = np.zeros((21, 3))
    hand[[5, 9, 13], 0] = 10.0  # knuckles to the RIGHT of the wrist (hand points right)
    assert hand_rotation(hand) == pytest.approx(np.pi / 2)
    hand[:, :2] = 0.0
    hand[[5, 9, 13], 1] = -10.0  # knuckles above the wrist (upright hand)
    assert hand_rotation(hand) == pytest.approx(0.0)
    # in the crop, "up" must point from the wrist towards the knuckles
    hand[:, :2] = np.array([[100.0, 100.0]])
    hand[[5, 9, 13], :2] = [[160.0, 100.0]] * 3
    hand[[1, 2, 3, 6, 10, 14, 17, 18], :2] = [[130.0, 90.0]] * 8
    sq = square_from_landmarks(hand)
    m = sq.image_to_crop_matrix(224)
    wrist, knuckle = (np.array([[100.0, 100.0], [160.0, 100.0]]) @ m[:, :2].T + m[:, 2])
    assert knuckle[1] < wrist[1] and abs(knuckle[0] - wrist[0]) < 1e-6


# -- decoding -------------------------------------------------------------------------
def test_decoder_recognises_outputs_by_range_and_order():
    screen = np.linspace(0, 200, 63)
    world = np.linspace(-0.05, 0.05, 63)
    outputs = {"fc_world": world, "fc_presence": np.array([0.93]), "fc_handedness": np.array([0.2]), "fc_screen": screen}
    landmarks, presence = HandLandmarkDecoder()(outputs)
    np.testing.assert_allclose(landmarks.ravel(), screen)
    assert presence == pytest.approx(0.93)
    _, p = HandLandmarkDecoder()({"a": screen, "b": np.array([3.0])})  # a raw logit
    assert p == pytest.approx(1 / (1 + np.exp(-3.0)))
    mapped = HandLandmarkDecoder(output_map={"landmarks": "fc_screen", "presence": "fc_handedness"})(outputs)
    assert mapped[1] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        HandLandmarkDecoder()({"a": world, "b": world, "c": np.array([0.5])})


# -- whole detector with the same network on the CPU ---------------------------------------
@pytest.fixture(scope="module")
def litert_runner(cfg):
    pytest.importorskip("ai_edge_litert")
    task = Path(cfg["detector"]["mediapipe"]["model_path"])
    if not task.exists() or not TEST_IMAGE.exists():
        pytest.skip("run `handpredict download-models mediapipe test-assets`")
    from litert_runner import LiteRTLandmarkRunner

    return LiteRTLandmarkRunner(task)


def test_hailo_tracking_follows_a_moving_hand(cfg, litert_runner):
    import cv2

    from handpredict.synthetic import ReachSpec, reach_position

    hand = cv2.imread(str(TEST_IMAGE))
    scale = 240 / hand.shape[0]
    hand = cv2.resize(hand, None, fx=scale, fy=scale)
    hh, hw = hand.shape[:2]
    fps = 60.0
    t = np.arange(90) / fps
    spec = ReachSpec(np.array([400.0, 380.0]), np.array([800.0, 330.0]), np.array([800.0, 330.0]), 0.3, 0.8, 0.0, 0.0)
    path = reach_position(spec, t)
    detector = HailoHandLandmarkDetector(cfg, runner=litert_runner)
    detector.set_search_region(path[0], 360)  # the calibrated start region
    found, offsets = 0, []
    for i, (cx, cy) in enumerate(path):
        canvas = np.full((720, 1280, 3), 90, np.uint8)
        x0, y0 = int(round(cx - hw / 2)), int(round(cy - hh / 2))
        canvas[y0 : y0 + hh, x0 : x0 + hw] = hand
        result = detector.detect(CapturedFrame(i, int(t[i] * 1e9), 0, canvas, "test"))
        if result.detected:
            found += 1
            offsets.append([result.x_px - cx, result.y_px - cy])
    assert found >= 85  # tracked in (almost) every frame
    offsets = np.asarray(offsets)
    # the knuckle is fixed on the pasted photo, so its offset from the path must stay constant
    assert np.std(offsets, axis=0).max() < 3.0
    assert detector.n_reacquired == 1  # found once at the start region, then tracked


def test_hailo_detector_agrees_with_mediapipe_on_a_photo(cfg, litert_runner):
    import cv2

    from handpredict.landmarks.mediapipe_detector import MediaPipeHandDetector

    image = cv2.imread(str(TEST_IMAGE))
    frame = CapturedFrame(0, 1_000_000, 0, image, "test")
    with MediaPipeHandDetector(cfg) as mp_det:
        reference = mp_det.detect(frame)
    detector = HailoHandLandmarkDetector(cfg, runner=litert_runner)
    detector.track = square_from_landmarks(reference.landmarks_px)  # start from MediaPipe's hand box
    first = detector.detect(frame)
    second = detector.detect(CapturedFrame(1, 2_000_000, 0, image, "test"))  # now tracking its own box
    for result in (first, second):
        assert result.detected
        assert np.hypot(result.x_px - reference.x_px, result.y_px - reference.y_px) < 6.0  # px


def test_missing_hailo_falls_back_to_mediapipe(cfg, caplog):
    pytest.importorskip("mediapipe")
    if not Path(cfg["detector"]["mediapipe"]["model_path"]).exists():
        pytest.skip("MediaPipe model not downloaded")
    from handpredict.landmarks import make_detector

    try:
        import hailo_platform  # noqa: F401

        pytest.skip("a Hailo runtime is installed here")
    except ImportError:
        pass
    with caplog.at_level("WARNING"):
        detector = make_detector(cfg, "hailo")
    assert detector.name == "mediapipe" and detector.fallback_from == "hailo"
    assert detector.describe()["fallback_from"] == "hailo"
    assert "Hailo device not found; falling back to CPU detector" in caplog.text
    detector.close()
