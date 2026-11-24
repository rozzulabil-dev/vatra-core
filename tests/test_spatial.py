"""Unit tests for spatial zone geometry and monitoring."""

from __future__ import annotations

import unittest

from vatra.spatial import (
    SpatialZoneMonitor,
    make_rect_polygon,
    point_in_polygon,
    transform_polygon,
)
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

    def test_make_rect_polygon_and_transform_polygon(self) -> None:
        rect = make_rect_polygon(100.0, 100.0, 40.0, 20.0, angle_deg=90.0)
        xs = [pt[0] for pt in rect]
        ys = [pt[1] for pt in rect]
        # 40x20 rotated 90 degrees becomes 20 wide x 40 tall around (100, 100)
        self.assertAlmostEqual(max(xs) - min(xs), 20.0, places=2)
        self.assertAlmostEqual(max(ys) - min(ys), 40.0, places=2)

        scaled_moved = transform_polygon(
            make_rect_polygon(100.0, 100.0, 40.0, 20.0),
            dx=50.0,
            dy=-20.0,
            scale_x=2.0,
            scale_y=1.5,
        )
        sxs = [pt[0] for pt in scaled_moved]
        sys_pts = [pt[1] for pt in scaled_moved]
        self.assertAlmostEqual((min(sxs) + max(sxs)) / 2.0, 150.0, places=2)
        self.assertAlmostEqual((min(sys_pts) + max(sys_pts)) / 2.0, 80.0, places=2)
        self.assertAlmostEqual(max(sxs) - min(sxs), 80.0, places=2)
        self.assertAlmostEqual(max(sys_pts) - min(sys_pts), 30.0, places=2)

    def test_bottom_center_anchor_accuracy(self) -> None:
        # Zone covers lower floor area y=[250, 400]; person's center is at y=200 (outside),
        # while feet (bottom_center) are at (200, 300) (inside).
        poly = [(100.0, 250.0), (300.0, 250.0), (300.0, 400.0), (100.0, 400.0)]
        packet = FramePacket(frame_id=1, timestamp_ms=33.3, width=640, height=480)
        det = Detection("person", 0.90, BoundingBox(170.0, 100.0, 230.0, 300.0))

        center_zone = SpatialZone("z-center", "Center Zone", poly, ("person",), anchor="center")
        foot_zone = SpatialZone("z-foot", "Foot Zone", poly, ("person",), anchor="bottom_center")

        self.assertEqual(SpatialZoneMonitor([center_zone]).evaluate(packet, [det]), [])
        foot_events = SpatialZoneMonitor([foot_zone]).evaluate(packet, [det])
        self.assertEqual(len(foot_events), 1)
        self.assertEqual(foot_events[0].zone_id, "z-foot")


if __name__ == "__main__":
    unittest.main()
