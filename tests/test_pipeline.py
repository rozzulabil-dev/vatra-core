"""End-to-end tests for EdgeVisionPipeline and CLI commands."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import unittest

from vatra.cli import main as cli_main
from vatra.detectors import MockDetector
from vatra.pipeline import EdgeVisionPipeline
from vatra.sources import SyntheticSource
from vatra.spatial import SpatialZoneMonitor
from vatra.types import SpatialZone, ZoneEvent


class PipelineAndCliTests(unittest.TestCase):
    def test_pipeline_end_to_end_with_synthetic_source(self) -> None:
        source = SyntheticSource(width=640, height=480, max_frames=20)
        detector = MockDetector(confidence_threshold=0.5)
        zone = SpatialZone(
            zone_id="center-roi",
            name="Center ROI",
            polygon=[(160.0, 120.0), (480.0, 120.0), (480.0, 360.0), (160.0, 360.0)],
            watched_labels=("person",),
        )
        monitor = SpatialZoneMonitor([zone])
        captured: list[ZoneEvent] = []

        pipeline = EdgeVisionPipeline(
            source=source,
            detector=detector,
            zone_monitor=monitor,
            on_event=captured.append,
        )
        stats = pipeline.run()

        self.assertEqual(stats.frames_processed, 20)
        self.assertGreater(stats.total_detections, 0)
        self.assertGreater(stats.total_zone_events, 0)
        self.assertEqual(stats.total_zone_events, len(captured))
        self.assertGreater(stats.avg_fps, 0.0)

    def test_cli_info_and_benchmark_json(self) -> None:
        info_out = io.StringIO()
        with contextlib.redirect_stdout(info_out):
            code = cli_main(["info", "--json"])
        self.assertEqual(code, 0)
        info_payload = json.loads(info_out.getvalue())
        self.assertEqual(info_payload["package"], "vatra-core")
        self.assertIn("packages", info_payload)

        bench_out = io.StringIO()
        with contextlib.redirect_stdout(bench_out):
            code = cli_main(["benchmark", "--frames", "10", "--json"])
        self.assertEqual(code, 0)
        bench_payload = json.loads(bench_out.getvalue())
        self.assertEqual(bench_payload["frames_processed"], 10)
        self.assertGreater(bench_payload["total_detections"], 0)
        self.assertGreater(bench_payload["total_zone_events"], 0)

    def test_cli_run_with_yaml_config_json(self) -> None:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default_pipeline.yaml"
        run_out = io.StringIO()
        with contextlib.redirect_stdout(run_out):
            code = cli_main(["run", "--config", str(cfg_path), "--max-frames", "10", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(run_out.getvalue())
        self.assertEqual(payload["stats"]["frames_processed"], 10)
        self.assertGreater(payload["events_count"], 0)


if __name__ == "__main__":
    unittest.main()
