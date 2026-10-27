"""Frame sources: Raspberry Pi camera, USB webcam, video file replay and synthetic data."""

from __future__ import annotations

from typing import Mapping, Sequence

from .base import CapturedFrame, FrameSource, SourceUnavailableError, count_dropped_frames


def make_source(
    cfg: Mapping,
    kind: str | None = None,
    video_path: str | None = None,
    timestamps_ns: Sequence[int] | None = None,
    seed: int | None = None,
    render: bool = False,
) -> FrameSource:
    """Create the frame source named in ``cfg['camera']['source']`` (or ``kind``)."""
    kind = kind or cfg["camera"]["source"]
    if kind == "picamera2":
        from .picamera2_source import Picamera2Source

        return Picamera2Source(cfg)
    if kind == "opencv":
        from .opencv_source import OpenCVCameraSource

        return OpenCVCameraSource(cfg)
    if kind == "video":
        from .video_source import VideoFileSource

        if not video_path:
            raise ValueError("the video source needs a video file path (--video)")
        return VideoFileSource(video_path, timestamps_ns=timestamps_ns, fps=cfg["camera"]["fps"])
    if kind == "synthetic":
        from .synthetic_source import SyntheticSource

        return SyntheticSource(cfg, seed=seed, render=render)
    raise ValueError(f"Unknown camera source {kind!r} (use picamera2, opencv, video or synthetic)")


__all__ = ["CapturedFrame", "FrameSource", "SourceUnavailableError", "count_dropped_frames", "make_source"]
