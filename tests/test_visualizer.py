"""Unit tests for OpenCV FrameVisualizer overlay rendering and video export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vatra.types import BoundingBox, Detection, FramePacket, SpatialZone, ZoneEvent
from vatra.visualizer import FrameVisualizer


class VisualizerTests(unittest.TestCase):
    def test_render_synthetic_frame_and_save_mp4(self) -> None:
        zone = SpatialZone(
            zone_id="z1",
            name="Test Perimeter",
            polygon=[(160.0, 120.0), (480.0, 120.0), (480.0, 360.0), (160.0, 360.0)],
            watched_labels=("person",),
        )
        packet = FramePacket(frame_id=0, timestamp_ms=0.0, width=640, height=480, data=None)
        det = Detection("person", 0.92, BoundingBox(260.0, 160.0, 360.0, 320.0))
        event = ZoneEvent(
            frame_id=0,
            timestamp_ms=0.0,
            zone_id="z1",
            zone_name="Test Perimeter",
            detections=[det],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_video = Path(tmp_dir) / "annotated.mp4"
            vis = FrameVisualizer(zones=[zone], show=False, save_path=str(out_video))
            try:
                canvas, keep_running = vis.render(packet, [det], [event])
            finally:
                vis.close()

            self.assertTrue(keep_running)
            self.assertEqual(canvas.shape, (480, 640, 3))
            self.assertTrue(out_video.exists())
            self.assertGreater(out_video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
