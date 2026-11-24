"""Unit tests for JSONL alert logging and rate-limited HTTP webhook dispatch."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vatra.alerts import JsonlAlertSink, WebhookAlertSink, zone_event_to_dict
from vatra.types import BoundingBox, Detection, ZoneEvent


def _sample_event(frame_id: int = 1, zone_id: str = "z1") -> ZoneEvent:
    return ZoneEvent(
        frame_id=frame_id,
        timestamp_ms=33.3,
        zone_id=zone_id,
        zone_name="Restricted Zone",
        detections=[
            Detection(
                label="person",
                confidence=0.92,
                bbox=BoundingBox(100.0, 120.0, 180.0, 280.0),
                class_id=0,
                track_id=1,
            )
        ],
    )


class AlertSinksTests(unittest.TestCase):
    def test_jsonl_alert_sink_appends_valid_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "logs" / "events.jsonl"
            sink = JsonlAlertSink(log_path)
            sink.emit(_sample_event(frame_id=3))
            sink.emit(_sample_event(frame_id=4))

            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["frame_id"], 3)
            self.assertEqual(first["zone_id"], "z1")
            self.assertEqual(first["detections"][0]["track_id"], 1)

    def test_webhook_alert_sink_rate_limiting(self) -> None:
        sent_calls: list[tuple[str, dict, float]] = []

        def fake_sender(url: str, payload: bytes, timeout: float) -> None:
            sent_calls.append((url, json.loads(payload.decode("utf-8")), timeout))

        sink = WebhookAlertSink(
            url="https://example.local/hook",
            cooldown_seconds=60.0,
            timeout_seconds=2.5,
            sender=fake_sender,
        )

        self.assertTrue(sink.emit(_sample_event(frame_id=1, zone_id="z1")))
        # Suppressed within cooldown for same zone
        self.assertFalse(sink.emit(_sample_event(frame_id=2, zone_id="z1")))
        # Different zone still sends immediately
        self.assertTrue(sink.emit(_sample_event(frame_id=2, zone_id="z2")))

        self.assertEqual(len(sent_calls), 2)
        self.assertEqual(sent_calls[0][0], "https://example.local/hook")
        self.assertEqual(sent_calls[0][1]["detections"][0]["track_id"], 1)
        self.assertEqual(sent_calls[0][2], 2.5)
        self.assertEqual(zone_event_to_dict(_sample_event(frame_id=5))["frame_id"], 5)


if __name__ == "__main__":
    unittest.main()
