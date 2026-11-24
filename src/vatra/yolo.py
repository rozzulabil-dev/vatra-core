"""YOLO ONNX adapter, class-aware NMS, and COCO class labels for vatra-core."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from vatra.types import BoundingBox, Detection, FramePacket

# 80 standard COCO object categories grouped by domain
COCO_LABELS: tuple[str, ...] = (
    # People & vehicles (0-8)
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    # Street & outdoor (9-13)
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    # Animals (14-23)
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # Personal gear (24-28)
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    # Sports (29-38)
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    # Dining & kitchenware (39-45)
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    # Food (46-55)
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    # Furniture (56-61)
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    # Electronics (62-67)
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    # Appliances & household (68-79)
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union_area = box_a.area + box_b.area - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


def non_max_suppression(
    detections: Sequence[Detection],
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Perform class-aware greedy Non-Maximum Suppression (NMS)."""
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        grouped[det.class_id].append(det)

    kept: list[Detection] = []
    for class_dets in grouped.values():
        sorted_dets = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
        selected: list[Detection] = []
        for candidate in sorted_dets:
            if all(compute_iou(candidate.bbox, prev.bbox) <= iou_threshold for prev in selected):
                selected.append(candidate)
        kept.extend(selected)

    return sorted(kept, key=lambda d: d.confidence, reverse=True)


def _clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    max_w: float,
    max_h: float,
) -> BoundingBox:
    cx1 = max(0.0, min(max_w, x1))
    cy1 = max(0.0, min(max_h, y1))
    cx2 = max(cx1, min(max_w, x2))
    cy2 = max(cy1, min(max_h, y2))
    return BoundingBox(x1=cx1, y1=cy1, x2=cx2, y2=cy2)


def decode_yolo_output(
    raw_output: Any,
    orig_width: int,
    orig_height: int,
    input_width: int = 640,
    input_height: int = 640,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    labels: Sequence[str] = COCO_LABELS,
) -> list[Detection]:
    """Decode YOLOv8/v11 [1, 4+C, N] or pre-decoded [1, N, 6] ONNX outputs."""
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for decode_yolo_output. "
            "Install it with: pip install 'vatra-core[vision]'"
        ) from exc

    arr = np.asarray(raw_output, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return []

    scale_x = float(orig_width) / float(max(input_width, 1))
    scale_y = float(orig_height) / float(max(input_height, 1))
    max_w = float(orig_width)
    max_h = float(orig_height)

    # Handle pre-decoded [N, 6] format: [x1, y1, x2, y2, score, class_id]
    expected_channels = 4 + len(labels)
    if arr.shape[1] == 6 and arr.shape[0] != expected_channels:
        candidates: list[Detection] = []
        for row in arr:
            score = float(row[4])
            if score < confidence_threshold:
                continue
            class_id = int(row[5])
            label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            bbox = _clamp_box(
                float(row[0]) * scale_x,
                float(row[1]) * scale_y,
                float(row[2]) * scale_x,
                float(row[3]) * scale_y,
                max_w,
                max_h,
            )
            candidates.append(Detection(label=label, confidence=score, bbox=bbox, class_id=class_id))
        return non_max_suppression(candidates, iou_threshold=iou_threshold)

    # Handle YOLOv8/v11 channels-first [4+C, N] -> transpose to [N, 4+C]
    if arr.shape[0] == expected_channels or (arr.shape[0] >= 5 and arr.shape[1] != expected_channels):
        preds = arr.T
    else:
        preds = arr

    if preds.shape[1] < 5 or preds.shape[0] == 0:
        return []

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    row_indices = np.arange(preds.shape[0])
    confidences = class_scores[row_indices, class_ids]

    keep_mask = confidences >= confidence_threshold
    if not np.any(keep_mask):
        return []

    filtered_boxes = boxes_xywh[keep_mask]
    filtered_scores = confidences[keep_mask]
    filtered_ids = class_ids[keep_mask]

    candidates = []
    for box, score, cid in zip(filtered_boxes, filtered_scores, filtered_ids):
        cx, cy, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        half_w = bw / 2.0
        half_h = bh / 2.0
        bbox = _clamp_box(
            (cx - half_w) * scale_x,
            (cy - half_h) * scale_y,
            (cx + half_w) * scale_x,
            (cy + half_h) * scale_y,
            max_w,
            max_h,
        )
        class_id = int(cid)
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        candidates.append(
            Detection(
                label=label,
                confidence=float(score),
                bbox=bbox,
                class_id=class_id,
            )
        )

    return non_max_suppression(candidates, iou_threshold=iou_threshold)


class YoloOnnxDetector:
    """YOLOv8/v11 ONNX Runtime detector prioritizing TensorRT and CUDA providers."""

    name: str = "yolo-onnx"

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        prefer_tensorrt: bool = True,
        input_size: tuple[int, int] = (640, 640),
        labels: Sequence[str] | None = None,
    ) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime, numpy, and opencv-python-headless are required for YoloOnnxDetector. "
                "Install them with: pip install 'vatra-core[vision,onnx]' "
                "or 'vatra-core[vision,cuda]' for NVIDIA GPU support."
            ) from exc

        self._cv2 = cv2
        self._np = np
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.labels: tuple[str, ...] = tuple(labels) if labels is not None else COCO_LABELS

        available = set(ort.get_available_providers())
        preferred_order = (
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            if prefer_tensorrt
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        providers = [p for p in preferred_order if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(model_path, providers=providers)
        self.active_providers: list[str] = list(self._session.get_providers())
        self._input_name: str = self._session.get_inputs()[0].name

    def _preprocess(self, frame_bgr: Any) -> Any:
        cv2 = self._cv2
        np = self._np
        in_w, in_h = self.input_size
        resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        return np.ascontiguousarray(chw[np.newaxis, ...])

    def detect(self, packet: FramePacket) -> list[Detection]:
        if packet.data is None:
            return []

        tensor = self._preprocess(packet.data)
        outputs = self._session.run(None, {self._input_name: tensor})
        if not outputs:
            return []

        return decode_yolo_output(
            raw_output=outputs[0],
            orig_width=packet.width,
            orig_height=packet.height,
            input_width=self.input_size[0],
            input_height=self.input_size[1],
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            labels=self.labels,
        )
