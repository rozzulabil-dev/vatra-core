"""Vatra.ai - Open-source Physical AI & Edge Vision platform by Vantara Intech."""

from vatra.detectors import (
    DetectorBackend,
    MockDetector,
    OnnxRuntimeDetector,
    probe_runtime_environment,
)
from vatra.pipeline import EdgeVisionPipeline, PipelineStats
from vatra.sources import FrameSource, SyntheticSource, VideoStreamSource
from vatra.spatial import SpatialZoneMonitor, point_in_polygon
from vatra.types import BoundingBox, Detection, FramePacket, SpatialZone, ZoneEvent

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BoundingBox",
    "Detection",
    "DetectorBackend",
    "EdgeVisionPipeline",
    "FramePacket",
    "FrameSource",
    "MockDetector",
    "OnnxRuntimeDetector",
    "PipelineStats",
    "SpatialZone",
    "SpatialZoneMonitor",
    "SyntheticSource",
    "VideoStreamSource",
    "ZoneEvent",
    "point_in_polygon",
    "probe_runtime_environment",
]
