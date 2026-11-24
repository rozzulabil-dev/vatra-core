"""Command-line interface for vatra-core."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Sequence

from vatra import __version__
from vatra.config import (
    DetectorConfig,
    PipelineConfig,
    SourceConfig,
    build_pipeline_from_config,
    load_pipeline_config,
)
from vatra.detectors import MockDetector, probe_runtime_environment
from vatra.pipeline import EdgeVisionPipeline
from vatra.sources import SyntheticSource
from vatra.spatial import SpatialZoneMonitor
from vatra.types import Detection, FramePacket, SpatialZone, ZoneEvent
from vatra.visualizer import FrameVisualizer


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


def _resolve_run_config(args: argparse.Namespace) -> PipelineConfig:
    if args.config:
        cfg = load_pipeline_config(args.config)
        if args.source is not None:
            cfg.source.uri = args.source
        if args.max_frames is not None:
            cfg.source.max_frames = args.max_frames
        if args.backend is not None:
            cfg.detector.backend = args.backend
        if args.model is not None:
            cfg.detector.model_path = args.model
        if args.confidence is not None:
            cfg.detector.confidence_threshold = args.confidence
        if args.iou is not None:
            cfg.detector.iou_threshold = args.iou
        if args.no_tensorrt:
            cfg.detector.prefer_tensorrt = False
        return cfg

    return PipelineConfig(
        source=SourceConfig(
            uri=args.source if args.source is not None else "synthetic",
            max_frames=args.max_frames,
        ),
        detector=DetectorConfig(
            backend=args.backend if args.backend is not None else "mock",
            model_path=args.model,
            confidence_threshold=args.confidence if args.confidence is not None else 0.5,
            iou_threshold=args.iou if args.iou is not None else 0.45,
            prefer_tensorrt=not args.no_tensorrt,
        ),
        zones=[_default_zone()],
    )


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _resolve_run_config(args)
    if cfg.detector.backend.lower() in ("onnx", "yolo-onnx") and not cfg.detector.model_path:
        print(
            f"Error: --model is required when --backend {cfg.detector.backend} is selected.",
            file=sys.stderr,
        )
        return 2

    recorded_events: list[ZoneEvent] = []

    def handle_event(event: ZoneEvent) -> None:
        recorded_events.append(event)
        if not args.json:
            labels = ", ".join(f"{d.label}({d.confidence:.2f})" for d in event.detections)
            print(
                f"[frame {event.frame_id:04d}] Zone '{event.zone_name}' triggered: {labels}"
            )

    visualizer: FrameVisualizer | None = None
    on_frame = None
    if args.show or args.save:
        visualizer = FrameVisualizer(
            zones=cfg.zones,
            show=args.show,
            save_path=args.save,
        )

        def handle_frame(
            packet: FramePacket,
            detections: list[Detection],
            events: list[ZoneEvent],
        ) -> bool:
            _, keep_running = visualizer.render(packet, detections, events)
            return keep_running

        on_frame = handle_frame

    pipeline = build_pipeline_from_config(
        cfg,
        on_event=handle_event,
        on_frame=on_frame,
    )

    try:
        stats = pipeline.run()
    finally:
        if visualizer is not None:
            visualizer.close()

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
        "--config",
        default=None,
        help="Path to YAML pipeline configuration file (CLI flags override config values).",
    )
    run_parser.add_argument(
        "--source",
        default=None,
        help="Input source: 'synthetic', webcam index (e.g. 0), file path, or rtsp:// URL.",
    )
    run_parser.add_argument(
        "--backend",
        choices=["mock", "onnx", "yolo-onnx"],
        default=None,
        help="Detection backend.",
    )
    run_parser.add_argument("--model", default=None, help="Path to ONNX model file.")
    run_parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Minimum confidence threshold.",
    )
    run_parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold for YOLO NMS (default: 0.45).",
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
    run_parser.add_argument(
        "--show",
        action="store_true",
        help="Display live OpenCV overlay window (press q or Esc to quit).",
    )
    run_parser.add_argument(
        "--save",
        default=None,
        help="Save annotated output video to an .mp4 file.",
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
