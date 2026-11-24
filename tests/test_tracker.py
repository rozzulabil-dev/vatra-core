"""Unit tests for lightweight IoU multi-object tracking."""

from __future__ import annotations

import unittest

from vatra.tracker import IouTracker
from vatra.types import BoundingBox, Detection


class TrackerTests(unittest.TestCase):
    def test_persistent_track_id_across_frames_and_expiry(self) -> None:
        tracker = IouTracker(iou_threshold=0.3, max_missed=2)

        # Frame 1: two distinct detections receive IDs 1 and 2
        frame1 = [
            Detection("person", 0.90, BoundingBox(100.0, 100.0, 180.0, 260.0), class_id=0),
            Detection("car", 0.85, BoundingBox(400.0, 200.0, 520.0, 300.0), class_id=2),
        ]
        out1 = tracker.update(frame1)
        self.assertEqual([d.track_id for d in out1], [1, 2])
        self.assertEqual(tracker.active_tracks_count, 2)

        # Frame 2: both boxes move slightly -> preserve IDs 1 and 2
        frame2 = [
            Detection("person", 0.91, BoundingBox(112.0, 104.0, 192.0, 264.0), class_id=0),
            Detection("car", 0.87, BoundingBox(410.0, 204.0, 530.0, 304.0), class_id=2),
        ]
        out2 = tracker.update(frame2)
        self.assertEqual([d.track_id for d in out2], [1, 2])

        # Frame 3..5: 3 missed frames (> max_missed=2) -> tracks expire
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.active_tracks_count, 0)

        # New detection after expiry gets fresh ID 3
        out_after = tracker.update(
            [Detection("person", 0.92, BoundingBox(112.0, 104.0, 192.0, 264.0), class_id=0)]
        )
        self.assertEqual([d.track_id for d in out_after], [3])


if __name__ == "__main__":
    unittest.main()
