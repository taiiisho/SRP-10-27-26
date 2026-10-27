"""Region-of-interest (ROI) maths for running a hand-landmark network on a crop.

The Hailo model zoo's ``hand_landmark_lite`` is MediaPipe's hand-landmark
network: it does not search the whole image, it expects a 224 x 224 crop with
the hand in the middle, upright (wrist at the bottom, fingers up). MediaPipe
solves this by building the crop for the next frame from the landmarks of the
current one. This module reproduces that logic (MediaPipe's
HandLandmarksToRectCalculator + RectTransformationCalculator, scale 2.0,
shift -0.1 along the hand axis, square crop) in plain pixel coordinates:

    crop pixel q  ->  image pixel p:   p = centre + (side / N) * R(angle) @ (q - N/2)

with R the 2-D rotation matrix (image y axis points down). The inverse maps
image pixels into the crop, which is what ``cv2.warpAffine`` needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Landmarks MediaPipe uses for the hand box: wrist and the two lower joints of
# each finger, which move less than the fingertips.
ROI_LANDMARKS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP = 0, 5, 9, 13


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def normalize_radians(angle: float) -> float:
    return float(angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi)))


@dataclass(frozen=True)
class RotatedSquare:
    """A square crop region: centre (px), side length (px) and rotation (radians)."""

    center: tuple[float, float]
    side: float
    angle: float = 0.0

    def crop_to_image_matrix(self, n: int) -> np.ndarray:
        """2x3 affine matrix taking crop pixels (0..n) to image pixels."""
        a = (self.side / n) * _rotation(self.angle)
        t = np.asarray(self.center, dtype=float) - a @ np.array([n / 2.0, n / 2.0])
        return np.hstack([a, t[:, None]])

    def image_to_crop_matrix(self, n: int) -> np.ndarray:
        """2x3 affine matrix taking image pixels to crop pixels (for cv2.warpAffine)."""
        a = (n / self.side) * _rotation(-self.angle)
        t = np.array([n / 2.0, n / 2.0]) - a @ np.asarray(self.center, dtype=float)
        return np.hstack([a, t[:, None]])

    def crop(self, image: np.ndarray, n: int) -> np.ndarray:
        import cv2

        return cv2.warpAffine(
            image, self.image_to_crop_matrix(n), (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    def to_image(self, points_crop: np.ndarray, n: int) -> np.ndarray:
        """Map (k, 2) crop-pixel points back to image pixels."""
        m = self.crop_to_image_matrix(n)
        pts = np.asarray(points_crop, dtype=float)
        return pts @ m[:, :2].T + m[:, 2]


def hand_rotation(landmarks_px: np.ndarray) -> float:
    """Angle that turns the hand upright (MediaPipe's ComputeRotation), radians."""
    x0, y0 = landmarks_px[WRIST, :2]
    mid = (landmarks_px[INDEX_MCP, :2] + landmarks_px[RING_MCP, :2]) / 2.0
    x1, y1 = (mid + landmarks_px[MIDDLE_MCP, :2]) / 2.0
    return normalize_radians(np.pi / 2.0 - np.arctan2(-(y1 - y0), x1 - x0))


def square_from_landmarks(landmarks_px: np.ndarray, scale: float = 2.0, shift_y: float = -0.1) -> RotatedSquare:
    """The crop for the NEXT frame, built from this frame's 21 landmarks (pixels)."""
    pts = np.asarray(landmarks_px, dtype=float)[ROI_LANDMARKS, :2]
    angle = hand_rotation(np.asarray(landmarks_px, dtype=float))
    centre_aa = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    # Bounding box in the hand's own (upright) frame.
    local = (pts - centre_aa) @ _rotation(-angle).T
    lo, hi = local.min(axis=0), local.max(axis=0)
    centre_local = (lo + hi) / 2.0
    width, height = hi - lo
    centre = centre_aa + _rotation(angle) @ centre_local
    # Shift along the hand axis (towards the fingers for negative shift_y).
    centre = centre + _rotation(angle) @ np.array([0.0, shift_y * height])
    side = max(width, height) * scale
    return RotatedSquare((float(centre[0]), float(centre[1])), float(side), float(angle))
