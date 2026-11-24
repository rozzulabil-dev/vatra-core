"""Unit tests for YAML pipeline configuration loader and builder."""

from __future__ import annotations

import unittest
from pathlib import Path

from vatra.config import build_pipeline_from_config, load_pipeline_config


class ConfigTests(unittest.TestCase):
    def test_load_and_run_default_pipeline_config(self) -> None:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default_pipeline.yaml"
        cfg = load_pipeline_config(cfg_path)

        self.assertEqual(cfg.source.uri, "synthetic")
        self.assertEqual(cfg.detector.backend, "mock")
        self.assertEqual(cfg.detector.iou_threshold, 0.45)
        self.assertEqual(cfg.detector.input_size, (640, 640))
        self.assertEqual(len(cfg.zones), 1)
        self.assertEqual(cfg.zones[0].zone_id, "safety-perimeter-1")

        cfg.source.max_frames = 12
        pipeline = build_pipeline_from_config(cfg)
        stats = pipeline.run()

        self.assertEqual(stats.frames_processed, 12)
        self.assertGreater(stats.total_detections, 0)
        self.assertGreater(stats.total_zone_events, 0)


if __name__ == "__main__":
    unittest.main()
