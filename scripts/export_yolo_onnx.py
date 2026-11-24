"""Helper script to export a YOLOv8/v11 ONNX model into models/ for vatra-core."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a YOLO PyTorch checkpoint (e.g., yolov8n.pt) to ONNX."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model name or .pt path (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Destination directory for exported .onnx file (default: models).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square export input resolution (default: 640).",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print(
            "Ultralytics is not installed in this environment.\n"
            "Install it inside your venv to export YOLO weights automatically:\n"
            "  pip install ultralytics\n"
            f"Then run:\n  python scripts/export_yolo_onnx.py --model {args.model}\n"
            "Or place any compatible YOLOv8/v11 .onnx model directly into models/.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.model)
    exported_path = Path(yolo.export(format="onnx", imgsz=args.imgsz, simplify=True))
    target_path = out_dir / exported_path.name
    if exported_path.resolve() != target_path.resolve():
        shutil.move(str(exported_path), str(target_path))

    print(f"Exported ONNX model to: {target_path}")
    print(
        f"Test live webcam dashboard with:\n"
        f"  vatra serve --source 0 --backend yolo-onnx --model {target_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
