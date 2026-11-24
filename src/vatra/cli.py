"""Command-line interface for vatra-core."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Sequence

from vatra import __version__
from vatra.detectors import (
    DetectorBackend,
    MockDetector,
    OnnxRuntimeDetector,
    probe_runtime_environment,
)
from vatra.pipeline import EdgeVisionPipeline
from vatra.sources import FrameSource, SyntheticSource, VideoStreamSource
from vatra.spatial import SpatialZoneMonitor
from vatra.types import SpatialZone, ZoneEvent


def _default_zone(width: int = 640, height: int = 480) -> SpatialZone:
    return SpatialZone(
        zone_id="safety-perimeter-1",
        name="Restricted Machinery Perimeter",
        polygon=[
            (width * 0.25, height * 0.25),
            (width * 0.75, height * 0.25),
            (width * 0.75, height * 0.75),
            (width * 0.25, height * 0.75),
        ],
        watched_labels=("person", "forklift"),
    )


def _cmd_info(as_json: bool) -> int:
    env = probe_runtime_environment()
    payload = {
        "package": "vatra-core",
        "version": __version__,
        **env,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"vatra-core v{__version__}")
    print(f"Python:   {env['python_version']}")
    print(f"Platform: {env['platform']}")
    print("Packages:")
    for pkg, available in env["packages"].items():
        status = "installed" if available else "not installed"
        print(f"  - {pkg}: {status}")
    providers = ", ".join(env["onnx_providers"]) if env["onnx_providers"] else "none"
    print(f"ONNX Providers:     {providers}")
    print(f"CUDA Available:     {env['cuda_available']}")
    print(f"TensorRT Available: {env['tensorrt_available']}")
    return 0


def _cmd_benchmark(frames: int, width: int, height: int, as_json: bool) -> int:
    source = SyntheticSource(width=width, height=height, max_frames=frames)
    detector = MockDetector(confidence_threshold=0.5)
    monitor = SpatialZoneMonitor([_default_zone(width=width, height=height)])

    pipeline = EdgeVisionPipeline(
        source=source,
        detector=detector,
        zone_monitor=monitor,
    )
    stats = pipeline.run()

    if as_json:
        print(json.dumps(asdict(stats), indent=2))
        return 0

    print(f"vatra-core v{__version__} Benchmark ({width}x{height}, {frames} frames)")
    print(f"  Frames processed:  {stats.frames_processed}")
    print(f"  Total detections:  {stats.total_detections}")
    print(f"  Zone events:       {stats.total_zone_events}")
    print(f"  Elapsed time:      {stats.elapsed_seconds:.4f}s")
    print(f"  Average FPS:       {stats.avg_fps:.2f}")
    print(f"  Average latency:   {stats.avg_latency_ms:.4f} ms/frame")
    return 0


def _parse_source(raw_source: str, max_frames: int | None) -> FrameSource:
    if raw_source == "synthetic":
        return SyntheticSource(max_frames=max_frames if max_frames is not None else 30)
    uri: int | str = int(raw_source) if raw_source.isdigit() else raw_source
    return VideoStreamSource(uri=uri, max_frames=max_frames)


def _cmd_run(args: argparse.Namespace) -> int:
    source = _parse_source(args.source, args.max_frames)

    detector: DetectorBackend
    if args.backend == "mock":
        detector = MockDetector(confidence_threshold=args.confidence)
    else:
        if not args.model:
            print("Error: --model is required when --backend onnx is selected.", file=sys.stderr)
            return 2
        detector = OnnxRuntimeDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            prefer_tensorrt=not args.no_tensorrt,
        )

    monitor = SpatialZoneMonitor([_default_zone()])
    recorded_events: list[ZoneEvent] = []

    def handle_event(event: ZoneEvent) -> None:
        recorded_events.append(event)
        if not args.json:
            labels = ", ".join(f"{d.label}({d.confidence:.2f})" for d in event.detections)
            print(
                f"[frame {event.frame_id:04d}] Zone '{event.zone_name}' triggered: {labels}"
            )

    pipeline = EdgeVisionPipeline(
        source=source,
        detector=detector,
        zone_monitor=monitor,
        on_event=handle_event,
    )
    stats = pipeline.run()

    if args.json:
        print(
            json.dumps(
                {
                    "stats": asdict(stats),
                    "events_count": len(recorded_events),
                },
                indent=2,
            )
        )
    else:
        print(
            f"Done: {stats.frames_processed} frames | "
            f"{stats.avg_fps:.2f} FPS | "
            f"{stats.total_zone_events} zone events"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vatra",
        description="Vatra.ai - Open-source Physical AI & Edge Vision CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser(
        "info",
        help="Show runtime environment and available NVIDIA/ONNX acceleration.",
    )
    info_parser.add_argument("--json", action="store_true", help="Output as JSON.")

    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run a zero-dependency synthetic pipeline benchmark.",
    )
    bench_parser.add_argument("--frames", type=int, default=30, help="Number of frames.")
    bench_parser.add_argument("--width", type=int, default=640, help="Frame width.")
    bench_parser.add_argument("--height", type=int, default=480, help="Frame height.")
    bench_parser.add_argument("--json", action="store_true", help="Output as JSON.")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the Edge Vision pipeline on synthetic, camera, or RTSP input.",
    )
    run_parser.add_argument(
        "--source",
        default="synthetic",
        help="Input source: 'synthetic', webcam index (e.g. 0), file path, or rtsp:// URL.",
    )
    run_parser.add_argument(
        "--backend",
        choices=["mock", "onnx"],
        default="mock",
        help="Detection backend.",
    )
    run_parser.add_argument("--model", default=None, help="Path to ONNX model file.")
    run_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold.",
    )
    run_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit.",
    )
    run_parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="Disable TensorrtExecutionProvider preference.",
    )
    run_parser.add_argument("--json", action="store_true", help="Output as JSON.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info(as_json=args.json)
    if args.command == "benchmark":
        return _cmd_benchmark(
            frames=args.frames,
            width=args.width,
            height=args.height,
            as_json=args.json,
        )
    if args.command == "run":
        return _cmd_run(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
