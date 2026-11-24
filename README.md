# Vatra.ai (`vatra-core`)

**Open-source Physical AI & Edge Vision platform by Vantara Intech (*Vantara Infrastruktur Teknologi*).**

`vatra-core` makes real-world computer vision, polygonal spatial zone monitoring, and sensor-to-AI pipelines lightweight, modular, and free for developers, researchers, and industries globally.

---

## Why Vatra.ai?

Deploying Physical AI in real environments—such as workplace safety perimeters, industrial visual inspection, and logistics monitoring—often requires heavy compute and brittle glue code. **Vatra.ai** focuses on:

- **Zero-Dependency Core:** Pure Python standard library core (`vatra info`, `vatra benchmark`, spatial polygon engine, and pipeline orchestration) runs anywhere out of the box.
- **Edge-First Acceleration:** Optional ONNX Runtime backend automatically prioritizes **NVIDIA TensorRT** (`TensorrtExecutionProvider`) and **CUDA** (`CUDAExecutionProvider`), falling back cleanly to CPU.
- **Modular Ingestion:** Unified `FrameSource` interface supporting synthetic benchmarks, webcams, video files, and RTSP camera streams.
- **Spatial Zone Intelligence:** Built-in ray-casting `SpatialZoneMonitor` to emit structured `ZoneEvent`s when watched object classes enter polygonal regions of interest.

---

## Architecture

```text
[ Synthetic / Webcam / RTSP ] ---> [ FramePacket ] ---> [ Mock / ONNX + TensorRT Detector ] ---> [ SpatialZoneMonitor ] ---> [ ZoneEvent ]
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

Install optional vision and inference extras depending on your target hardware:

```bash
# Camera/RTSP ingestion + CPU ONNX Runtime
pip install -e ".[vision,onnx,yaml]"

# Camera/RTSP ingestion + NVIDIA CUDA / TensorRT ONNX Runtime
pip install -e ".[vision,cuda,yaml]"
```

Run inference on a webcam, video file, or RTSP stream with an ONNX model:

```bash
vatra run --source 0 --backend onnx --model path/to/detector.onnx
```

---

## Project Structure

```text
vatra-core/
├── configs/
│   └── default_pipeline.yaml
├── src/
│   └── vatra/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── detectors.py
│       ├── pipeline.py
│       ├── sources.py
│       ├── spatial.py
│       └── types.py
├── tests/
│   ├── test_pipeline.py
│   └── test_spatial.py
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
- [ ] Pre-configured YOLOv8/YOLOv10 model adapters & class label presets
- [ ] Synthetic data & simulation bridge (NVIDIA Omniverse / Isaac Sim compatible)
- [ ] Public edge benchmarks on NVIDIA Jetson Orin

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request as we expand the Physical AI modules.

## License

Distributed under the **Apache 2.0 License**. See `LICENSE` for details.

---
Maintained by **Vantara Intech** (*Vantara Infrastruktur Teknologi*).
