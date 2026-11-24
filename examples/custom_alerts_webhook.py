"""Example: Wiring JSONL logging and rate-limited HTTP webhooks into vatra-core."""

from __future__ import annotations

import os

from vatra import (
    EdgeVisionPipeline,
    IouTracker,
    JsonlAlertSink,
    MockDetector,
    SpatialZone,
    SpatialZoneMonitor,
    SyntheticSource,
    WebhookAlertSink,
    ZoneEvent,
)


def main() -> None:
    zone = SpatialZone(
        zone_id="dock-gate-1",
        name="Loading Dock Gate",
        polygon=[(160.0, 120.0), (480.0, 120.0), (480.0, 360.0), (160.0, 360.0)],
        watched_labels=("person",),
    )

    jsonl_sink = JsonlAlertSink("logs/example_events.jsonl")
    webhook_url = os.environ.get("VATRA_WEBHOOK_URL")
    webhook_sink = (
        WebhookAlertSink(url=webhook_url, cooldown_seconds=5.0)
        if webhook_url
        else None
    )

    def dispatch_event(event: ZoneEvent) -> None:
        jsonl_sink.emit(event)
        if webhook_sink is not None:
            try:
                sent = webhook_sink.emit(event)
                status = "sent" if sent else "suppressed by cooldown"
                print(f"[frame {event.frame_id:04d}] Webhook {status}")
            except Exception as exc:
                print(f"[frame {event.frame_id:04d}] Webhook error: {exc}")

    pipeline = EdgeVisionPipeline(
        source=SyntheticSource(width=640, height=480, max_frames=20),
        detector=MockDetector(confidence_threshold=0.5),
        tracker=IouTracker(iou_threshold=0.3, max_missed=10),
        zone_monitor=SpatialZoneMonitor([zone]),
        on_event=dispatch_event,
    )
    stats = pipeline.run()
    print(f"Logged {stats.total_zone_events} zone events to {jsonl_sink.path}")


if __name__ == "__main__":
    main()
