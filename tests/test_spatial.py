"""Unit tests for spatial zone geometry and monitoring."""

from __future__ import annotations

import unittest

from vatra.spatial import SpatialZoneMonitor, point_in_polygon
from vatra.types import BoundingBox, Detection, FramePacket, SpatialZone


class SpatialZoneTests(unittest.TestCase):
    def test_bounding_box_geometry(self) -> None:
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=100.0)
        self.assertEqual(bbox.width, 40.0)
        self.assertEqual(bbox.height, 80.0)
        self.assertEqual(bbox.area, 3200.0)
        self.assertEqual(bbox.center, (30.0, 60.0))

    def test_point_in_polygon(self) -> None:
        square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        self.assertTrue(point_in_polygon((50.0, 50.0), square))
        self.assertFalse(point_in_polygon((150.0, 50.0), square))
        self.assertFalse(point_in_polygon((50.0, 50.0), [(0.0, 0.0), (1.0, 1.0)]))

    def test_zone_monitor_filters_labels_and_bounds(self) -> None:
        zone = SpatialZone(
            zone_id="z1",
            name="Restricted Zone",
            polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
            watched_labels=("person",),
        )
        monitor = SpatialZoneMonitor([zone])
        packet = FramePacket(frame_id=1, timestamp_ms=33.3, width=640, height=480)

        detections = [
            # Inside polygon and watched label -> triggers
            Detection("person", 0.95, BoundingBox(150.0, 150.0, 210.0, 250.0)),
            # Inside polygon but unwatched label -> ignored
            Detection("pallet", 0.90, BoundingBox(180.0, 180.0, 220.0, 220.0)),
            # Watched label but outside polygon -> ignored
            Detection("person", 0.88, BoundingBox(10.0, 10.0, 50.0, 80.0)),
        ]

        events = monitor.evaluate(packet, detections)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].zone_id, "z1")
        self.assertEqual(len(events[0].detections), 1)
        self.assertEqual(events[0].detections[0].label, "person")


if __name__ == "__main__":
    unittest.main()
