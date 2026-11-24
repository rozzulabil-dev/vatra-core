# Vatra.ai (`vatra-core`)

**Open-source Physical AI & Edge Vision platform by Vantara Intech (*Vantara Infrastruktur Teknologi*).**

`vatra-core` makes real-world computer vision, multi-object tracking, zone dwell-time analytics, live local web telemetry, and sensor-to-AI pipelines lightweight, modular, and free for developers, researchers, and industries globally.

---

## Why Vatra.ai?

Deploying Physical AI in real environments—such as workplace safety perimeters, industrial visual inspection, and logistics monitoring—often requires heavy compute and brittle glue code. **Vatra.ai** focuses on:

- **Zero-Dependency Core:** Pure Python standard library core (`vatra info`, `vatra benchmark`, `IouTracker`, `ZoneAnalyticsTracker`, `JsonlAlertSink`, `WebhookAlertSink`, `VatraDashboardServer`, and spatial polygon engine) runs anywhere out of the box.
- **Edge-First Acceleration:** ONNX Runtime and `YoloOnnxDetector` backends automatically prioritize **NVIDIA TensorRT** (`TensorrtExecutionProvider`), **CUDA** (`CUDAExecutionProvider`), and **DirectML** (`DmlExecutionProvider`), falling back cleanly to CPU.
- **YOLO ONNX + Letterbox + NMS + Tracking:** Aspect-ratio-preserving `letterbox_image` preprocessing, YOLOv8/v11 ONNX decoding, class-aware Non-Maximum Suppression (`non_max_suppression`), 80 COCO class labels, and persistent `track_id` assignment (`--track`).
- **Zone Analytics & Live Dashboard (`v0.4.0`):** Computes active occupancy, unique entry counts, and average/max dwell time (`ZoneAnalyticsTracker`), with a built-in MJPEG + JSON local web dashboard (`vatra serve`).
- **YAML Config, Alerts & Visualizer:** Declarative `configs/default_pipeline.yaml` runner (`vatra run --config ...`) with JSONL event logging (`--log-events`), rate-limited HTTP webhooks (`--webhook`), OpenCV live preview (`--show`), and `.mp4` annotation export (`--save`).

---

## Architecture

```text
[ Synthetic / Webcam / RTSP ] ---> [ Mock / ONNX / YOLO-ONNX ] ---> [ IouTracker ] ---> [ SpatialZoneMonitor + ZoneAnalytics ] ---> [ Alerts / Visualizer / vatra serve ]
```

---

## Quickstart

Clone the repository and install in editable mode (core requires Python `>=3.10` with zero external dependencies):

```bash
git clone https://github.com/rozzulabil-dev/vatra-core.git
cd vatra-core
pip install -e .
```

Inspect detected runtime & NVIDIA acceleration providers:

```bash
vatra info
```

Run the zero-dependency synthetic pipeline benchmark:

```bash
vatra benchmark --frames 30
```

*(You can also run directly without installing via `PYTHONPATH=src python -m vatra info` on Linux/macOS or `$env:PYTHONPATH="src"; python -m vatra info` on PowerShell.)*

---

## Optional NVIDIA / Edge Acceleration

Install optional vision, YAML, and inference extras depending on your target hardware:

```bash
# Camera/RTSP ingestion + YAML config + CPU ONNX Runtime
pip install -e ".[vision,onnx,yaml]"

# Camera/RTSP ingestion + YAML config + NVIDIA CUDA / TensorRT ONNX Runtime
pip install -e ".[vision,cuda,yaml]"
```

Run from YAML config, launch the live browser dashboard (`vatra serve`), or deploy on a live webcam/RTSP feed:

```bash
# Run using YAML configuration with persistent track IDs, zone analytics, and JSONL logging
vatra run --config configs/default_pipeline.yaml --track --log-events logs/events.jsonl --max-frames 30

# Start the local web dashboard at http://127.0.0.1:8080 (MJPEG stream + Zone Analytics)
vatra serve --config configs/default_pipeline.yaml

# Export YOLOv8n to ONNX into models/ (requires ultralytics) and run live webcam dashboard
python scripts/export_yolo_onnx.py --model yolov8n.pt
vatra serve --source 0 --backend yolo-onnx --model models/yolov8n.onnx

# Run YOLOv8/v11 ONNX model on webcam with live HUD window, webhook alerts, and MP4 export
vatra run --source 0 --backend yolo-onnx --model models/yolov8n.onnx --track --webhook https://example.local/alerts --show --save out.mp4
```

---

## Docker

Run the live dashboard in a container (CPU default or NVIDIA CUDA):

```bash
# CPU container
docker build -t vatra-core .
docker run --rm -p 8080:8080 vatra-core

# NVIDIA CUDA / TensorRT container
docker build --build-arg EXTRAS=vision,cuda,yaml -t vatra-core:cuda .
docker run --rm --gpus all -p 8080:8080 vatra-core:cuda
```

---

## Project Structure

```text
vatra-core/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── default_pipeline.yaml
├── docs/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── examples/
│   ├── basic_python_api.py
│   └── custom_alerts_webhook.py
├── scripts/
│   └── export_yolo_onnx.py
├── src/
│   └── vatra/
│       ├── __init__.py
│       ├── __main__.py
│       ├── alerts.py
│       ├── analytics.py
│       ├── cli.py
│       ├── config.py
│       ├── detectors.py
│       ├── pipeline.py
│       ├── server.py
│       ├── sources.py
│       ├── spatial.py
│       ├── tracker.py
│       ├── types.py
│       ├── visualizer.py
│       └── yolo.py
├── tests/
│   ├── test_alerts.py
│   ├── test_analytics.py
│   ├── test_config.py
│   ├── test_pipeline.py
│   ├── test_server.py
│   ├── test_spatial.py
│   ├── test_tracker.py
│   ├── test_visualizer.py
│   └── test_yolo.py
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Running Tests

```bash
python -m unittest discover -s tests -v
```

---

## Roadmap

- [x] Project initialization & open-source specification
- [x] Core Edge Vision pipeline, spatial zone monitor, and CLI (`v0.1.0`)
- [x] ONNX Runtime backend with TensorRT / CUDA provider priority
- [x] YAML pipeline config loader, YOLO ONNX adapter + NMS + COCO labels, and OpenCV visualizer/recorder (`v0.2.0`)
- [x] Multi-object IoU tracking, JSONL/Webhook alert dispatchers, and GitHub Actions CI (`v0.3.0`)
- [x] Per-zone dwell-time & unique-entry analytics + live local web dashboard `vatra serve` (`v0.4.0`)
- [x] Aspect-ratio letterbox YOLO preprocessing, Windows GPU DirectML/CUDA/TensorRT priority & ONNX export helper (`v0.5.0`)
- [ ] Synthetic data & simulation bridge (NVIDIA Omniverse / Isaac Sim compatible)
- [ ] Public edge benchmarks on NVIDIA Jetson Orin

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request as we expand the Physical AI modules.

## License

Distributed under the **Apache 2.0 License**. See `LICENSE` for details.

---
Maintained by **Vantara Intech** (*Vantara Infrastruktur Teknologi*).
