# Vatra.ai (`vatra-core`)

**Open-source Physical AI & Edge Vision platform by Vantara Intech (*Vantara Infrastruktur Teknologi*).**

`vatra-core` makes real-world computer vision, polygonal spatial zone monitoring, and sensor-to-AI pipelines lightweight, modular, and free for developers, researchers, and industries globally.

---

## Why Vatra.ai?

Deploying Physical AI in real environments—such as workplace safety perimeters, industrial visual inspection, and logistics monitoring—often requires heavy compute and brittle glue code. **Vatra.ai** focuses on:

- **Zero-Dependency Core:** Pure Python standard library core (`vatra info`, `vatra benchmark`, spatial polygon engine, and pipeline orchestration) runs anywhere out of the box.
- **Edge-First Acceleration:** ONNX Runtime and `YoloOnnxDetector` backends automatically prioritize **NVIDIA TensorRT** (`TensorrtExecutionProvider`) and **CUDA** (`CUDAExecutionProvider`), falling back cleanly to CPU.
- **YOLO ONNX + NMS Ready:** Built-in YOLOv8/v11 ONNX decoding, class-aware Non-Maximum Suppression (`non_max_suppression`), and 80 COCO class labels.
- **YAML Config & Live Visualizer:** Declarative `configs/default_pipeline.yaml` runner (`vatra run --config ...`) with optional OpenCV live preview (`--show`) and `.mp4` annotation export (`--save`).
- **Spatial Zone Intelligence:** Built-in ray-casting `SpatialZoneMonitor` to emit structured `ZoneEvent`s when watched object classes enter polygonal regions of interest.

---

## Architecture

```text
[ Synthetic / Webcam / RTSP ] ---> [ FramePacket ] ---> [ Mock / ONNX / YOLO-ONNX + NMS ] ---> [ SpatialZoneMonitor ] ---> [ ZoneEvent & Visualizer ]
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

Run from YAML config or directly against a webcam, video file, or RTSP stream with a YOLOv8/v11 ONNX model:

```bash
# Run using YAML configuration
vatra run --config configs/default_pipeline.yaml --max-frames 30

# Run YOLOv8/v11 ONNX model on webcam with live HUD window and MP4 export
vatra run --source 0 --backend yolo-onnx --model yolov8n.onnx --show --save out.mp4
```

---

## Project Structure

```text
vatra-core/
├── configs/
│   └── default_pipeline.yaml
├── docs/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── src/
│   └── vatra/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── detectors.py
│       ├── pipeline.py
│       ├── sources.py
│       ├── spatial.py
│       ├── types.py
│       ├── visualizer.py
│       └── yolo.py
├── tests/
│   ├── test_config.py
│   ├── test_pipeline.py
│   ├── test_spatial.py
│   ├── test_visualizer.py
│   └── test_yolo.py
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
- [ ] Synthetic data & simulation bridge (NVIDIA Omniverse / Isaac Sim compatible)
- [ ] Public edge benchmarks on NVIDIA Jetson Orin

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request as we expand the Physical AI modules.

## License

Distributed under the **Apache 2.0 License**. See `LICENSE` for details.

---
Maintained by **Vantara Intech** (*Vantara Infrastruktur Teknologi*).
