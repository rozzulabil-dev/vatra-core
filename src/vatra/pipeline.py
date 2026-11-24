"""Real-time Edge Vision and Physical AI pipeline orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from vatra.detectors import DetectorBackend
from vatra.sources import FrameSource
from vatra.spatial import SpatialZoneMonitor
from vatra.tracker import IouTracker
from vatra.types import Detection, FramePacket, ZoneEvent


@dataclass(slots=True)
class PipelineStats:
    """Execution summary metrics produced after running an EdgeVisionPipeline."""

    frames_processed: int
    total_detections: int
    total_zone_events: int
    elapsed_seconds: float
    avg_fps: float
    avg_latency_ms: float


class EdgeVisionPipeline:
    """Coordinates frame ingestion, object detection, and spatial zone evaluation."""

    def __init__(
        self,
        source: FrameSource,
        detector: DetectorBackend,
        zone_monitor: SpatialZoneMonitor | None = None,
        on_event: Callable[[ZoneEvent], None] | None = None,
        on_frame: Callable[[FramePacket, list[Detection], list[ZoneEvent]], bool | None] | None = None,
        tracker: IouTracker | None = None,
    ) -> None:
        self.source = source
        self.detector = detector
        self.zone_monitor = zone_monitor
        self.on_event = on_event
        self.on_frame = on_frame
        self.tracker = tracker

    def run(self) -> PipelineStats:
        """Process all frames from the source and return throughput statistics."""
        frames_processed = 0
        total_detections = 0
        total_zone_events = 0
        total_frame_time = 0.0

        wall_start = time.perf_counter()
        try:
            for packet in self.source:
                frame_start = time.perf_counter()
                detections = self.detector.detect(packet)
                if self.tracker is not None:
                    detections = self.tracker.update(detections)
                total_detections += len(detections)

                events: list[ZoneEvent] = []
                if self.zone_monitor is not None:
                    events = self.zone_monitor.evaluate(packet, detections)
                    total_zone_events += len(events)
                    if self.on_event is not None:
                        for event in events:
                            self.on_event(event)

                keep_running = True
                if self.on_frame is not None:
                    result = self.on_frame(packet, detections, events)
                    if result is False:
                        keep_running = False

                total_frame_time += time.perf_counter() - frame_start
                frames_processed += 1
                if not keep_running:
                    break
        finally:
            self.source.close()

        elapsed_seconds = max(time.perf_counter() - wall_start, 1e-9)
        avg_fps = frames_processed / elapsed_seconds if frames_processed > 0 else 0.0
        avg_latency_ms = (
            (total_frame_time / frames_processed) * 1000.0
            if frames_processed > 0
            else 0.0
        )

        return PipelineStats(
            frames_processed=frames_processed,
            total_detections=total_detections,
            total_zone_events=total_zone_events,
            elapsed_seconds=round(elapsed_seconds, 6),
            avg_fps=round(avg_fps, 2),
            avg_latency_ms=round(avg_latency_ms, 4),
        )
