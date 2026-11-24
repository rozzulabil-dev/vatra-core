"""Core data structures for Vatra Edge Vision and Physical AI pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned 2D bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(slots=True)
class Detection:
    """Single object detection result produced by a detector backend."""

    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int = 0


@dataclass(slots=True)
class FramePacket:
    """Container representing one video or sensor frame entering the pipeline."""

    frame_id: int
    timestamp_ms: float
    width: int
    height: int
    data: Any = None
    source_id: str = "default"


@dataclass(slots=True)
class SpatialZone:
    """Polygonal region of interest monitored for Physical AI spatial events."""

    zone_id: str
    name: str
    polygon: list[tuple[float, float]]
    watched_labels: tuple[str, ...] = ()


@dataclass(slots=True)
class ZoneEvent:
    """Event emitted when one or more detections fall inside a SpatialZone."""

    frame_id: int
    timestamp_ms: float
    zone_id: str
    zone_name: str
    detections: list[Detection]
