"""Example: Running the vatra-core pipeline with tracking and zone analytics in Python."""

from __future__ import annotations

from vatra import (
    EdgeVisionPipeline,
    IouTracker,
    MockDetector,
    SpatialZone,
    SpatialZoneMonitor,
    SyntheticSource,
    ZoneAnalyticsTracker,
    ZoneEvent,
)


def main() -> None:
    zone = SpatialZone(
        zone_id="safety-perimeter-1",
        name="Restricted Machinery Perimeter",
        polygon=[(160.0, 120.0), (480.0, 120.0), (480.0, 360.0), (160.0, 360.0)],
        watched_labels=("person", "forklift"),
    )

    source = SyntheticSource(width=640, height=480, max_frames=20)
    detector = MockDetector(confidence_threshold=0.5)
    tracker = IouTracker(iou_threshold=0.3, max_missed=10)
    monitor = SpatialZoneMonitor([zone])
    analytics = ZoneAnalyticsTracker([zone])

    def handle_event(event: ZoneEvent) -> None:
        tags = ", ".join(f"{d.label}#{d.track_id}({d.confidence:.2f})" for d in event.detections)
        print(f"[frame {event.frame_id:04d}] Zone '{event.zone_name}': {tags}")

    pipeline = EdgeVisionPipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        zone_monitor=monitor,
        on_event=handle_event,
        on_frame=lambda packet, _dets, events: bool(analytics.update(packet, events) or True),
    )

    stats = pipeline.run()
    print(
        f"\nSummary: {stats.frames_processed} frames | "
        f"{stats.avg_fps:.2f} FPS | "
        f"{stats.total_zone_events} zone events"
    )
    for snap in analytics.summary().values():
        print(
            f"Analytics [{snap.zone_name}]: unique={snap.unique_entries}, "
            f"avg_dwell={snap.avg_dwell_seconds:.2f}s, "
            f"max_dwell={snap.max_dwell_seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
