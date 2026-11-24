"""Unit tests for per-zone occupancy, unique entry counts, and dwell-time analytics."""

from __future__ import annotations

import unittest

from vatra.analytics import ZoneAnalyticsTracker
from vatra.types import BoundingBox, Detection, FramePacket, SpatialZone, ZoneEvent


class ZoneAnalyticsTests(unittest.TestCase):
    def test_unique_entries_and_dwell_time(self) -> None:
        zone = SpatialZone(
            zone_id="z1",
            name="Loading Dock",
            polygon=[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)],
            watched_labels=("person",),
        )
        tracker = ZoneAnalyticsTracker([zone])

        # Frame 0 (t=0ms): track #1 enters
        p0 = FramePacket(frame_id=0, timestamp_ms=0.0, width=640, height=480)
        d1 = Detection("person", 0.9, BoundingBox(20.0, 20.0, 80.0, 120.0), track_id=1)
        snap0 = tracker.update(p0, [ZoneEvent(0, 0.0, "z1", "Loading Dock", [d1])])["z1"]
        self.assertEqual(snap0.active_count, 1)
        self.assertEqual(snap0.unique_entries, 1)
        self.assertEqual(snap0.max_dwell_seconds, 0.0)

        # Frame 1 (t=1500ms): track #1 stays + track #2 enters -> dwell #1 = 1.5s, #2 = 0.0s
        p1 = FramePacket(frame_id=1, timestamp_ms=1500.0, width=640, height=480)
        d2 = Detection("person", 0.88, BoundingBox(100.0, 40.0, 150.0, 140.0), track_id=2)
        snap1 = tracker.update(p1, [ZoneEvent(1, 1500.0, "z1", "Loading Dock", [d1, d2])])["z1"]
        self.assertEqual(snap1.active_count, 2)
        self.assertEqual(snap1.unique_entries, 2)
        self.assertAlmostEqual(snap1.max_dwell_seconds, 1.5, places=3)
        self.assertAlmostEqual(snap1.avg_dwell_seconds, 0.75, places=3)

        # Frame 2 (t=2000ms): zone empty -> active_count resets to 0, unique & dwell retained
        p2 = FramePacket(frame_id=2, timestamp_ms=2000.0, width=640, height=480)
        snap2 = tracker.update(p2, [])["z1"]
        self.assertEqual(snap2.active_count, 0)
        self.assertEqual(snap2.unique_entries, 2)
        self.assertAlmostEqual(snap2.max_dwell_seconds, 1.5, places=3)


if __name__ == "__main__":
    unittest.main()
