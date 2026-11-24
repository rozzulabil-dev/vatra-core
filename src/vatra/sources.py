"""Frame ingestion sources for synthetic benchmarks, cameras, and RTSP streams."""

from __future__ import annotations

import time
from typing import Any, Iterator, Protocol

from vatra.types import FramePacket


class FrameSource(Protocol):
    """Protocol implemented by all video and sensor frame sources."""

    def __iter__(self) -> Iterator[FramePacket]:
        ...

    def close(self) -> None:
        ...


class SyntheticSource:
    """Zero-dependency synthetic frame generator for testing and benchmarking."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        max_frames: int = 30,
        source_id: str = "synthetic",
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers.")
        if max_frames < 0:
            raise ValueError("max_frames must be >= 0.")
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.source_id = source_id
        self._closed = False

    def __iter__(self) -> Iterator[FramePacket]:
        for frame_id in range(self.max_frames):
            if self._closed:
                break
            yield FramePacket(
                frame_id=frame_id,
                timestamp_ms=round(frame_id * (1000.0 / 30.0), 3),
                width=self.width,
                height=self.height,
                data=None,
                source_id=self.source_id,
            )

    def close(self) -> None:
        self._closed = True


class VideoStreamSource:
    """OpenCV-backed video capture source for webcams, files, and RTSP streams."""

    def __init__(
        self,
        uri: int | str,
        max_frames: int | None = None,
        source_id: str | None = None,
    ) -> None:
        self.uri = uri
        self.max_frames = max_frames
        self.source_id = source_id or str(uri)
        self._cap: Any = None
        self._closed = False

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python-headless is required for VideoStreamSource. "
                "Install it with: pip install 'vatra-core[vision]'"
            ) from exc

        cap = cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open video source: {self.uri!r}")

        self._cap = cap
        frame_id = 0
        start_time = time.perf_counter()

        try:
            while not self._closed:
                if self.max_frames is not None and frame_id >= self.max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                height, width = frame.shape[:2]
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                yield FramePacket(
                    frame_id=frame_id,
                    timestamp_ms=round(elapsed_ms, 3),
                    width=int(width),
                    height=int(height),
                    data=frame,
                    source_id=self.source_id,
                )
                frame_id += 1
        finally:
            self.close()

    def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None
