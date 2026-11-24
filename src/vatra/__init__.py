"""Vatra.ai - Open-source Physical AI & Edge Vision platform by Vantara Intech."""

__version__ = "0.2.0"

from vatra.config import (
    DetectorConfig,
    PipelineConfig,
    SourceConfig,
    build_pipeline_from_config,
    load_pipeline_config,
)
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
from vatra.visualizer import FrameVisualizer
from vatra.yolo import (
    COCO_LABELS,
    YoloOnnxDetector,
    compute_iou,
    decode_yolo_output,
    non_max_suppression,
)

__all__ = [
    "__version__",
    "BoundingBox",
    "COCO_LABELS",
    "Detection",
    "DetectorBackend",
    "DetectorConfig",
    "EdgeVisionPipeline",
    "FramePacket",
    "FrameSource",
    "FrameVisualizer",
    "MockDetector",
    "OnnxRuntimeDetector",
    "PipelineConfig",
    "PipelineStats",
    "SourceConfig",
    "SpatialZone",
    "SpatialZoneMonitor",
    "SyntheticSource",
    "VideoStreamSource",
    "YoloOnnxDetector",
    "ZoneEvent",
    "build_pipeline_from_config",
    "compute_iou",
    "decode_yolo_output",
    "load_pipeline_config",
    "non_max_suppression",
    "point_in_polygon",
    "probe_runtime_environment",
]
