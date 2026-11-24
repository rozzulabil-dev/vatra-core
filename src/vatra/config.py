"""YAML pipeline configuration loader and builder for vatra-core."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vatra.detectors import DetectorBackend, MockDetector, OnnxRuntimeDetector
from vatra.pipeline import EdgeVisionPipeline
from vatra.sources import FrameSource, SyntheticSource, VideoStreamSource
from vatra.spatial import SpatialZoneMonitor
from vatra.tracker import IouTracker
from vatra.types import Detection, FramePacket, SpatialZone, ZoneEvent
from vatra.yolo import YoloOnnxDetector


@dataclass(slots=True)
class SourceConfig:
    """Configuration for pipeline frame ingestion."""

    uri: int | str = "synthetic"
    width: int = 640
    height: int = 480
    max_frames: int | None = 60


@dataclass(slots=True)
class DetectorConfig:
    """Configuration for the object detection backend."""

    backend: str = "mock"
    model_path: str | None = None
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    prefer_tensorrt: bool = True
    input_size: tuple[int, int] = (640, 640)
    labels: list[str] | None = None


@dataclass(slots=True)
class TrackerConfig:
    """Configuration for lightweight IoU multi-object tracking."""

    enabled: bool = False
    iou_threshold: float = 0.3
    max_missed: int = 10


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration parsed from YAML."""

    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    zones: list[SpatialZone] = field(default_factory=list)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate a PipelineConfig from a YAML file."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyyaml is required to load YAML configuration files. "
            "Install it with: pip install 'vatra-core[yaml]'"
        ) from exc

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Pipeline config file not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config root must be a YAML mapping.")

    src_raw: dict[str, Any] = raw.get("source") or {}
    if not isinstance(src_raw, dict):
        raise ValueError("'source' section must be a mapping.")
    source_cfg = SourceConfig(
        uri=src_raw.get("uri", "synthetic"),
        width=int(src_raw.get("width", 640)),
        height=int(src_raw.get("height", 480)),
        max_frames=int(src_raw["max_frames"]) if src_raw.get("max_frames") is not None else None,
    )

    det_raw: dict[str, Any] = raw.get("detector") or {}
    if not isinstance(det_raw, dict):
        raise ValueError("'detector' section must be a mapping.")
    raw_size = det_raw.get("input_size", [640, 640])
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        raise ValueError("'detector.input_size' must be a 2-element list [width, height].")
    raw_labels = det_raw.get("labels")
    detector_cfg = DetectorConfig(
        backend=str(det_raw.get("backend", "mock")),
        model_path=str(det_raw["model_path"]) if det_raw.get("model_path") else None,
        confidence_threshold=float(det_raw.get("confidence_threshold", 0.5)),
        iou_threshold=float(det_raw.get("iou_threshold", 0.45)),
        prefer_tensorrt=bool(det_raw.get("prefer_tensorrt", True)),
        input_size=(int(raw_size[0]), int(raw_size[1])),
        labels=[str(lbl) for lbl in raw_labels] if raw_labels is not None else None,
    )

    trk_raw: dict[str, Any] = raw.get("tracker") or {}
    if not isinstance(trk_raw, dict):
        raise ValueError("'tracker' section must be a mapping.")
    tracker_cfg = TrackerConfig(
        enabled=bool(trk_raw.get("enabled", False)),
        iou_threshold=float(trk_raw.get("iou_threshold", 0.3)),
        max_missed=int(trk_raw.get("max_missed", 10)),
    )

    zones_raw = raw.get("zones") or []
    if not isinstance(zones_raw, list):
        raise ValueError("'zones' section must be a list of spatial zones.")
    zones: list[SpatialZone] = []
    for idx, z in enumerate(zones_raw):
        if not isinstance(z, dict):
            raise ValueError(f"Zone at index {idx} must be a mapping.")
        polygon_raw = z.get("polygon") or []
        polygon = [(float(pt[0]), float(pt[1])) for pt in polygon_raw]
        watched = tuple(str(lbl) for lbl in (z.get("watched_labels") or ()))
        zones.append(
            SpatialZone(
                zone_id=str(z.get("zone_id", f"zone-{idx + 1}")),
                name=str(z.get("name", f"Zone {idx + 1}")),
                polygon=polygon,
                watched_labels=watched,
            )
        )

    return PipelineConfig(
        source=source_cfg,
        detector=detector_cfg,
        tracker=tracker_cfg,
        zones=zones,
    )


def build_source_from_config(source_cfg: SourceConfig) -> FrameSource:
    """Instantiate a FrameSource from SourceConfig."""
    if source_cfg.uri == "synthetic":
        return SyntheticSource(
            width=source_cfg.width,
            height=source_cfg.height,
            max_frames=source_cfg.max_frames if source_cfg.max_frames is not None else 30,
        )
    uri: int | str = source_cfg.uri
    if isinstance(uri, str) and uri.isdigit():
        uri = int(uri)
    return VideoStreamSource(uri=uri, max_frames=source_cfg.max_frames)


def build_detector_from_config(det_cfg: DetectorConfig) -> DetectorBackend:
    """Instantiate a DetectorBackend from DetectorConfig."""
    backend = det_cfg.backend.lower()
    if backend == "mock":
        return MockDetector(confidence_threshold=det_cfg.confidence_threshold)

    if backend in ("onnx", "yolo-onnx"):
        if not det_cfg.model_path:
            raise ValueError(f"'model_path' is required when backend is {backend!r}.")
        if backend == "onnx":
            return OnnxRuntimeDetector(
                model_path=det_cfg.model_path,
                confidence_threshold=det_cfg.confidence_threshold,
                prefer_tensorrt=det_cfg.prefer_tensorrt,
                labels=det_cfg.labels,
            )
        return YoloOnnxDetector(
            model_path=det_cfg.model_path,
            confidence_threshold=det_cfg.confidence_threshold,
            iou_threshold=det_cfg.iou_threshold,
            prefer_tensorrt=det_cfg.prefer_tensorrt,
            input_size=det_cfg.input_size,
            labels=det_cfg.labels,
        )

    raise ValueError(f"Unsupported detector backend: {det_cfg.backend!r}")


def build_pipeline_from_config(
    config: PipelineConfig,
    on_event: Callable[[ZoneEvent], None] | None = None,
    on_frame: Callable[[FramePacket, list[Detection], list[ZoneEvent]], bool | None] | None = None,
) -> EdgeVisionPipeline:
    """Build a runnable EdgeVisionPipeline from a PipelineConfig."""
    source = build_source_from_config(config.source)
    detector = build_detector_from_config(config.detector)
    tracker = (
        IouTracker(
            iou_threshold=config.tracker.iou_threshold,
            max_missed=config.tracker.max_missed,
        )
        if config.tracker.enabled
        else None
    )
    monitor = SpatialZoneMonitor(config.zones) if config.zones else None
    return EdgeVisionPipeline(
        source=source,
        detector=detector,
        zone_monitor=monitor,
        on_event=on_event,
        on_frame=on_frame,
        tracker=tracker,
    )
