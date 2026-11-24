"""Command-line interface for vatra-core."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict
from typing import Any, Sequence

from vatra import __version__
from vatra.alerts import JsonlAlertSink, WebhookAlertSink
from vatra.analytics import ZoneAnalyticsTracker
from vatra.config import (
    DetectorConfig,
    PipelineConfig,
    SourceConfig,
    TrackerConfig,
    build_pipeline_from_config,
    build_source_from_config,
    load_pipeline_config,
)
from vatra.detectors import MockDetector, probe_runtime_environment
from vatra.pipeline import EdgeVisionPipeline
from vatra.server import DashboardState, VatraDashboardServer
from vatra.sources import SyntheticSource
from vatra.spatial import SpatialZoneMonitor, make_rect_polygon
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
    print(f"DirectML Available: {env['directml_available']}")
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


def _resolve_pipeline_config(
    args: argparse.Namespace,
    *,
    default_track: bool = False,
) -> PipelineConfig:
    if args.config:
        cfg = load_pipeline_config(args.config)
        if args.source is not None:
            cfg.source.uri = args.source
        if getattr(args, "max_frames", None) is not None:
            cfg.source.max_frames = args.max_frames
        if args.backend is not None:
            cfg.detector.backend = args.backend
        if args.model is not None:
            cfg.detector.model_path = args.model
        if getattr(args, "confidence", None) is not None:
            cfg.detector.confidence_threshold = args.confidence
        if getattr(args, "iou", None) is not None:
            cfg.detector.iou_threshold = args.iou
        if getattr(args, "no_tensorrt", False):
            cfg.detector.prefer_tensorrt = False
        if getattr(args, "track", False):
            cfg.tracker.enabled = True
        if getattr(args, "no_track", False):
            cfg.tracker.enabled = False
        return cfg

    track_enabled = default_track
    if getattr(args, "track", False):
        track_enabled = True
    if getattr(args, "no_track", False):
        track_enabled = False

    return PipelineConfig(
        source=SourceConfig(
            uri=args.source if args.source is not None else "synthetic",
            max_frames=getattr(args, "max_frames", None),
        ),
        detector=DetectorConfig(
            backend=args.backend if args.backend is not None else "mock",
            model_path=args.model,
            confidence_threshold=(
                args.confidence if getattr(args, "confidence", None) is not None else 0.5
            ),
            iou_threshold=args.iou if getattr(args, "iou", None) is not None else 0.45,
            prefer_tensorrt=not getattr(args, "no_tensorrt", False),
        ),
        tracker=TrackerConfig(enabled=track_enabled),
        zones=[_default_zone()],
    )


def _format_detection_tag(det: Detection) -> str:
    if det.track_id is not None:
        return f"{det.label}#{det.track_id}({det.confidence:.2f})"
    return f"{det.label}({det.confidence:.2f})"


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _resolve_pipeline_config(args, default_track=False)
    if cfg.detector.backend.lower() in ("onnx", "yolo-onnx") and not cfg.detector.model_path:
        print(
            f"Error: --model is required when --backend {cfg.detector.backend} is selected.",
            file=sys.stderr,
        )
        return 2

    jsonl_sink = JsonlAlertSink(args.log_events) if args.log_events else None
    webhook_sink = (
        WebhookAlertSink(args.webhook, cooldown_seconds=args.webhook_cooldown)
        if args.webhook
        else None
    )
    analytics_tracker = ZoneAnalyticsTracker(cfg.zones)
    recorded_events: list[ZoneEvent] = []

    def handle_event(event: ZoneEvent) -> None:
        recorded_events.append(event)
        if jsonl_sink is not None:
            jsonl_sink.emit(event)
        if webhook_sink is not None:
            try:
                webhook_sink.emit(event)
            except Exception as exc:
                print(f"Warning: webhook dispatch failed: {exc}", file=sys.stderr)
        if not args.json:
            labels = ", ".join(_format_detection_tag(d) for d in event.detections)
            print(
                f"[frame {event.frame_id:04d}] Zone '{event.zone_name}' triggered: {labels}"
            )

    visualizer: FrameVisualizer | None = None
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
        analytics_tracker.update(packet, events)
        if visualizer is not None:
            _, keep_running = visualizer.render(packet, detections, events)
            return keep_running
        return True

    pipeline = build_pipeline_from_config(
        cfg,
        on_event=handle_event,
        on_frame=handle_frame,
    )

    try:
        stats = pipeline.run()
    finally:
        if visualizer is not None:
            visualizer.close()

    analytics_summary = analytics_tracker.summary()
    if args.json:
        print(
            json.dumps(
                {
                    "stats": asdict(stats),
                    "events_count": len(recorded_events),
                    "analytics": analytics_tracker.summary_dict(),
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
        if recorded_events:
            for snap in analytics_summary.values():
                print(
                    f"  Zone '{snap.zone_name}': unique={snap.unique_entries}, "
                    f"avg_dwell={snap.avg_dwell_seconds:.2f}s, "
                    f"max_dwell={snap.max_dwell_seconds:.2f}s"
                )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        print(
            "Error: opencv-python-headless is required for 'vatra serve'. "
            "Install it with: pip install 'vatra-core[vision]'",
            file=sys.stderr,
        )
        return 2

    cfg = _resolve_pipeline_config(args, default_track=True)
    if cfg.detector.backend.lower() in ("onnx", "yolo-onnx") and not cfg.detector.model_path:
        print(
            f"Error: --model is required when --backend {cfg.detector.backend} is selected.",
            file=sys.stderr,
        )
        return 2

    if not cfg.zones:
        cfg.zones = [_default_zone()]

    active_zone = cfg.zones[0]
    if len(active_zone.polygon) >= 3:
        xs = [pt[0] for pt in active_zone.polygon]
        ys = [pt[1] for pt in active_zone.polygon]
        init_cx = (min(xs) + max(xs)) / 2.0
        init_cy = (min(ys) + max(ys)) / 2.0
        init_w = max(40.0, max(xs) - min(xs))
        init_h = max(40.0, max(ys) - min(ys))
    else:
        init_cx, init_cy, init_w, init_h = 320.0, 240.0, 320.0, 240.0

    zone_lock = threading.Lock()
    initial_control: dict[str, Any] = {
        "zone_id": active_zone.zone_id,
        "zone_name": active_zone.name,
        "center_x": round(init_cx, 1),
        "center_y": round(init_cy, 1),
        "width": round(init_w, 1),
        "height": round(init_h, 1),
        "angle_deg": 0.0,
        "anchor": active_zone.anchor,
        "confidence": float(cfg.detector.confidence_threshold),
        "frame_width": int(cfg.source.width),
        "frame_height": int(cfg.source.height),
    }
    zone_control: dict[str, Any] = dict(initial_control)

    state = DashboardState(version=__version__)
    state.update(zone_control=zone_control)
    visualizer = FrameVisualizer(zones=cfg.zones, show=False, save_path=None)
    analytics_tracker = ZoneAnalyticsTracker(cfg.zones)
    frame_interval = 1.0 / max(args.fps, 1.0)
    loop_synthetic = cfg.source.uri == "synthetic" and args.max_frames is None

    total_frames = 0
    total_detections = 0
    total_events = 0
    start_wall = time.perf_counter()

    def handle_frame(
        packet: FramePacket,
        detections: list[Detection],
        events: list[ZoneEvent],
    ) -> bool:
        nonlocal total_frames, total_detections, total_events
        total_frames += 1
        total_detections += len(detections)
        total_events += len(events)

        updated_control: dict[str, Any] | None = None
        with zone_lock:
            if (
                packet.width != zone_control["frame_width"]
                or packet.height != zone_control["frame_height"]
            ):
                zone_control["frame_width"] = int(packet.width)
                zone_control["frame_height"] = int(packet.height)
                updated_control = dict(zone_control)

        analytics_tracker.update(packet, events)
        canvas, _ = visualizer.render(packet, detections, events)
        ok, encoded = cv2.imencode(".jpg", canvas)
        jpeg_bytes = bytes(encoded) if ok else None

        elapsed = max(time.perf_counter() - start_wall, 1e-6)
        state.update(
            jpeg_bytes=jpeg_bytes,
            stats={
                "frames_processed": total_frames,
                "total_detections": total_detections,
                "total_zone_events": total_events,
                "avg_fps": round(total_frames / elapsed, 2),
            },
            analytics=analytics_tracker.summary_dict(),
            events=events,
            zone_control=updated_control,
        )
        if cfg.source.uri == "synthetic":
            time.sleep(frame_interval)
        return True

    pipeline = build_pipeline_from_config(cfg, on_frame=handle_frame)
    if pipeline.tracker is not None:
        pipeline.tracker.smoothing_alpha = 0.65

    def handle_zone_update(payload: dict[str, Any]) -> dict[str, Any]:
        with zone_lock:
            if payload.get("reset"):
                fw = zone_control["frame_width"]
                fh = zone_control["frame_height"]
                zone_control.update(initial_control)
                zone_control["frame_width"] = fw
                zone_control["frame_height"] = fh
            else:
                if "center_x" in payload:
                    zone_control["center_x"] = float(payload["center_x"])
                if "center_y" in payload:
                    zone_control["center_y"] = float(payload["center_y"])
                if "width" in payload:
                    zone_control["width"] = max(20.0, float(payload["width"]))
                if "height" in payload:
                    zone_control["height"] = max(20.0, float(payload["height"]))
                if "angle_deg" in payload:
                    zone_control["angle_deg"] = max(-180.0, min(180.0, float(payload["angle_deg"])))
                if "anchor" in payload and payload["anchor"] in ("center", "bottom_center"):
                    zone_control["anchor"] = str(payload["anchor"])
                if "confidence" in payload:
                    zone_control["confidence"] = max(0.05, min(0.99, float(payload["confidence"])))

            active_zone.polygon = make_rect_polygon(
                center_x=zone_control["center_x"],
                center_y=zone_control["center_y"],
                width=zone_control["width"],
                height=zone_control["height"],
                angle_deg=zone_control["angle_deg"],
            )
            active_zone.anchor = zone_control["anchor"]
            if hasattr(pipeline.detector, "confidence_threshold"):
                pipeline.detector.confidence_threshold = zone_control["confidence"]
            return dict(zone_control)

    server = VatraDashboardServer(
        host=args.host,
        port=args.port,
        state=state,
        on_zone_update=handle_zone_update,
    )
    server.start_background()
    print(f"Vatra.ai Live Dashboard running at {server.url} (Press Ctrl+C to stop)")

    try:
        providers = getattr(pipeline.detector, "active_providers", None)
        if providers is not None:
            print(f"Detector: {pipeline.detector.name} | Providers: {', '.join(providers)}")
        else:
            print(f"Detector: {pipeline.detector.name}")

        while True:
            pipeline.run()
            if not loop_synthetic:
                break
            pipeline.source = build_source_from_config(cfg.source)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.close()
        server.stop()

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
        "--track",
        action="store_true",
        help="Enable IoU multi-object tracking across frames.",
    )
    run_parser.add_argument(
        "--log-events",
        default=None,
        help="Append triggered ZoneEvents to a JSONL file.",
    )
    run_parser.add_argument(
        "--webhook",
        default=None,
        help="HTTP endpoint URL to POST rate-limited JSON ZoneEvents.",
    )
    run_parser.add_argument(
        "--webhook-cooldown",
        type=float,
        default=5.0,
        help="Cooldown seconds per zone between webhook calls (default: 5.0).",
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

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the local Vatra.ai web dashboard with live MJPEG stream and zone analytics.",
    )
    serve_parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML pipeline configuration file.",
    )
    serve_parser.add_argument(
        "--source",
        default=None,
        help="Input source: 'synthetic', webcam index (e.g. 0), file path, or rtsp:// URL.",
    )
    serve_parser.add_argument(
        "--backend",
        choices=["mock", "onnx", "yolo-onnx"],
        default=None,
        help="Detection backend.",
    )
    serve_parser.add_argument("--model", default=None, help="Path to ONNX model file.")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the web dashboard (default: 127.0.0.1).",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port for the web dashboard (default: 8080).",
    )
    serve_parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target stream pace in frames per second (default: 15.0).",
    )
    serve_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit before exiting.",
    )
    serve_parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable IoU multi-object tracking.",
    )

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
    if args.command == "serve":
        return _cmd_serve(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
