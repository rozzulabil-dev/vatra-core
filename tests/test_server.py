"""Unit tests for the local Vatra.ai web dashboard server."""

from __future__ import annotations

import json
import unittest
import urllib.request

from vatra.server import DashboardState, VatraDashboardServer
from vatra.types import BoundingBox, Detection, ZoneEvent


class DashboardServerTests(unittest.TestCase):
    def test_dashboard_html_and_status_api(self) -> None:
        state = DashboardState(version="0.4.0")
        state.update(
            stats={
                "frames_processed": 12,
                "total_detections": 24,
                "total_zone_events": 4,
                "avg_fps": 30.5,
            },
            analytics={
                "z1": {
                    "zone_id": "z1",
                    "zone_name": "Safety Perimeter",
                    "active_count": 1,
                    "unique_entries": 2,
                    "avg_dwell_seconds": 1.25,
                    "max_dwell_seconds": 2.0,
                }
            },
            events=[
                ZoneEvent(
                    frame_id=12,
                    timestamp_ms=400.0,
                    zone_id="z1",
                    zone_name="Safety Perimeter",
                    detections=[
                        Detection(
                            "person",
                            0.92,
                            BoundingBox(100.0, 100.0, 180.0, 260.0),
                            track_id=1,
                        )
                    ],
                )
            ],
        )

        server = VatraDashboardServer(host="127.0.0.1", port=0, state=state)
        server.start_background()
        try:
            with urllib.request.urlopen(f"{server.url}/", timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("Vatra.ai", html)
                self.assertIn("/stream.mjpg", html)

            with urllib.request.urlopen(f"{server.url}/api/status", timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(payload["version"], "0.4.0")
                self.assertEqual(payload["stats"]["frames_processed"], 12)
                self.assertIn("z1", payload["analytics"])
                self.assertEqual(len(payload["recent_events"]), 1)
                self.assertEqual(payload["recent_events"][0]["detections"][0]["track_id"], 1)
        finally:
            server.stop()

    def test_post_zone_update_endpoint(self) -> None:
        state = DashboardState(version="0.6.0")
        received: list[dict] = []

        def on_zone_update(payload: dict) -> dict:
            received.append(payload)
            return {
                "center_x": payload.get("center_x", 320.0),
                "center_y": payload.get("center_y", 240.0),
                "width": payload.get("width", 200.0),
                "height": payload.get("height", 150.0),
                "angle_deg": payload.get("angle_deg", 25.0),
                "confidence": payload.get("confidence", 0.6),
                "anchor": payload.get("anchor", "bottom_center"),
                "frame_width": 640,
                "frame_height": 480,
            }

        server = VatraDashboardServer(
            host="127.0.0.1",
            port=0,
            state=state,
            on_zone_update=on_zone_update,
        )
        server.start_background()
        try:
            req = urllib.request.Request(
                f"{server.url}/api/zone",
                data=json.dumps(
                    {
                        "center_x": 400,
                        "center_y": 280,
                        "width": 220,
                        "height": 160,
                        "angle_deg": 25,
                        "confidence": 0.6,
                        "anchor": "bottom_center",
                    }
                ).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                updated = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(updated["center_x"], 400)
                self.assertEqual(updated["angle_deg"], 25)
                self.assertEqual(updated["anchor"], "bottom_center")
                self.assertEqual(len(received), 1)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
