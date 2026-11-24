"""Detector backends (mock and ONNX Runtime with CUDA/TensorRT priority)."""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any, Protocol

from vatra.types import BoundingBox, Detection, FramePacket


class DetectorBackend(Protocol):
    """Protocol implemented by all Vatra inference backends."""

    name: str

    def detect(self, packet: FramePacket) -> list[Detection]:
        ...


class MockDetector:
    """Deterministic synthetic detector for zero-hardware testing and benchmarking."""

    name: str = "mock"

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def detect(self, packet: FramePacket) -> list[Detection]:
        w = float(packet.width)
        h = float(packet.height)

        # Simulate a person moving horizontally across the center of the frame
        cycle_pos = (packet.frame_id % 20) / 19.0
        cx = w * (0.15 + 0.70 * cycle_pos)
        cy = h * 0.50
        box_w = w * 0.12
        box_h = h * 0.36

        candidates = [
            Detection(
                label="person",
                confidence=0.92,
                class_id=0,
                bbox=BoundingBox(
                    x1=round(cx - box_w / 2.0, 2),
                    y1=round(cy - box_h / 2.0, 2),
                    x2=round(cx + box_w / 2.0, 2),
                    y2=round(cy + box_h / 2.0, 2),
                ),
            ),
            Detection(
                label="pallet",
                confidence=0.84,
                class_id=1,
                bbox=BoundingBox(
                    x1=round(w * 0.05, 2),
                    y1=round(h * 0.70, 2),
                    x2=round(w * 0.18, 2),
                    y2=round(h * 0.90, 2),
                ),
            ),
        ]

        return [d for d in candidates if d.confidence >= self.confidence_threshold]


class OnnxRuntimeDetector:
    """ONNX Runtime object detector preferring TensorRT and CUDA when available."""

    name: str = "onnxruntime"

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        prefer_tensorrt: bool = True,
        labels: list[str] | None = None,
    ) -> None:
        try:
            import numpy as np  # type: ignore[import-not-found]
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime and numpy are required for OnnxRuntimeDetector. "
                "Install them with: pip install 'vatra-core[vision,onnx]' "
                "or 'vatra-core[vision,cuda]' for NVIDIA GPU support."
            ) from exc

        self._np = np
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.labels = labels or []

        available = set(ort.get_available_providers())
        preferred_order = (
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
            if prefer_tensorrt
            else ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        )
        providers = [p for p in preferred_order if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(model_path, providers=providers)
        self.active_providers: list[str] = list(self._session.get_providers())
        self._input_meta = self._session.get_inputs()[0]
        self._input_name: str = self._input_meta.name

    def _prepare_tensor(self, data: Any) -> Any:
        np = self._np
        arr = np.asarray(data)
        if arr.ndim == 3:
            # Convert HWC uint8/float image to NCHW float32 normalized to [0, 1]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        elif arr.ndim == 4:
            arr = arr.astype(np.float32)
        else:
            raise ValueError(
                f"Expected 3D (HWC) or 4D (NCHW) frame tensor, got shape {arr.shape}."
            )
        return np.ascontiguousarray(arr)

    def detect(self, packet: FramePacket) -> list[Detection]:
        if packet.data is None:
            return []

        tensor = self._prepare_tensor(packet.data)
        outputs = self._session.run(None, {self._input_name: tensor})
        if not outputs:
            return []

        raw = self._np.asarray(outputs[0])
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]

        detections: list[Detection] = []
        if raw.ndim != 2 or raw.shape[1] < 6:
            return detections

        for row in raw:
            score = float(row[4])
            if score < self.confidence_threshold:
                continue
            class_id = int(row[5])
            label = (
                self.labels[class_id]
                if 0 <= class_id < len(self.labels)
                else f"class_{class_id}"
            )
            detections.append(
                Detection(
                    label=label,
                    confidence=score,
                    class_id=class_id,
                    bbox=BoundingBox(
                        x1=float(row[0]),
                        y1=float(row[1]),
                        x2=float(row[2]),
                        y2=float(row[3]),
                    ),
                )
            )

        return detections


def probe_runtime_environment() -> dict[str, Any]:
    """Inspect Python runtime and installed Vision / NVIDIA acceleration packages."""
    packages = {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "opencv": importlib.util.find_spec("cv2") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "pyyaml": importlib.util.find_spec("yaml") is not None,
    }

    providers: list[str] = []
    if packages["onnxruntime"]:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            providers = list(ort.get_available_providers())
        except Exception:
            providers = []

    torch_cuda = False
    if packages["torch"]:
        try:
            import torch  # type: ignore[import-not-found]

            torch_cuda = bool(torch.cuda.is_available())
        except Exception:
            torch_cuda = False

    cuda_available = ("CUDAExecutionProvider" in providers) or torch_cuda
    directml_available = "DmlExecutionProvider" in providers
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "onnx_providers": providers,
        "cuda_available": cuda_available,
        "tensorrt_available": "TensorrtExecutionProvider" in providers,
        "directml_available": directml_available,
        "gpu_available": cuda_available or directml_available,
    }
