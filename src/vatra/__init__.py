"""Vatra.ai - Open-source Physical AI & Edge Vision platform by Vantara Intech."""

__version__ = "0.4.0"

from vatra.alerts import JsonlAlertSink, WebhookAlertSink, zone_event_to_dict
from vatra.analytics import ZoneAnalyticsSnapshot, ZoneAnalyticsTracker
from vatra.config import (
    DetectorConfig,
    PipelineConfig,
    SourceConfig,
    TrackerConfig,
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
from vatra.server import DashboardState, VatraDashboardServer
from vatra.sources import FrameSource, SyntheticSource, VideoStreamSource
from vatra.spatial import SpatialZoneMonitor, point_in_polygon
from vatra.tracker import IouTracker
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
    "DashboardState",
    "Detection",
    "DetectorBackend",
    "DetectorConfig",
    "EdgeVisionPipeline",
    "FramePacket",
    "FrameSource",
    "FrameVisualizer",
    "IouTracker",
    "JsonlAlertSink",
    "MockDetector",
    "OnnxRuntimeDetector",
    "PipelineConfig",
    "PipelineStats",
    "SourceConfig",
    "SpatialZone",
    "SpatialZoneMonitor",
    "SyntheticSource",
    "TrackerConfig",
    "VatraDashboardServer",
    "VideoStreamSource",
    "WebhookAlertSink",
    "YoloOnnxDetector",
    "ZoneAnalyticsSnapshot",
    "ZoneAnalyticsTracker",
    "ZoneEvent",
    "build_pipeline_from_config",
    "compute_iou",
    "decode_yolo_output",
    "load_pipeline_config",
    "non_max_suppression",
    "point_in_polygon",
    "probe_runtime_environment",
    "zone_event_to_dict",
]
