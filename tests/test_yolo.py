"""Unit tests for YOLO ONNX decoding, IoU, and class-aware NMS."""

from __future__ import annotations

import unittest

import numpy as np

from vatra.types import BoundingBox, Detection
from vatra.yolo import (
    COCO_LABELS,
    compute_iou,
    decode_yolo_output,
    non_max_suppression,
)


class YoloTests(unittest.TestCase):
    def test_coco_labels_count(self) -> None:
        self.assertEqual(len(COCO_LABELS), 80)
        self.assertEqual(COCO_LABELS[0], "person")
        self.assertEqual(COCO_LABELS[2], "car")

    def test_compute_iou(self) -> None:
        box = BoundingBox(0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(compute_iou(box, box), 1.0)

        disjoint = BoundingBox(20.0, 20.0, 30.0, 30.0)
        self.assertAlmostEqual(compute_iou(box, disjoint), 0.0)

        half_overlap = BoundingBox(5.0, 0.0, 15.0, 10.0)
        # Intersection = 50, Union = 100 + 100 - 50 = 150
        self.assertAlmostEqual(compute_iou(box, half_overlap), 50.0 / 150.0)

    def test_non_max_suppression_class_aware(self) -> None:
        dets = [
            Detection("person", 0.90, BoundingBox(10.0, 10.0, 110.0, 110.0), class_id=0),
            # High overlap, same class -> suppressed
            Detection("person", 0.75, BoundingBox(15.0, 15.0, 112.0, 112.0), class_id=0),
            # High overlap, different class -> kept
            Detection("bicycle", 0.80, BoundingBox(12.0, 12.0, 108.0, 108.0), class_id=1),
        ]
        kept = non_max_suppression(dets, iou_threshold=0.45)
        self.assertEqual(len(kept), 2)
        self.assertEqual([(d.label, d.confidence) for d in kept], [("person", 0.90), ("bicycle", 0.80)])

    def test_decode_yolo_output_yolov8_tensor(self) -> None:
        # Synthetic YOLOv8 [1, 84, 3] tensor in 640x640 input space
        raw = np.zeros((1, 84, 3), dtype=np.float32)

        # Box 0: person (class 0) centered at (320, 320), size (160, 320), conf=0.92
        raw[0, 0:4, 0] = [320.0, 320.0, 160.0, 320.0]
        raw[0, 4 + 0, 0] = 0.92

        # Box 1: duplicate overlapping person with lower conf=0.65 -> suppressed by NMS
        raw[0, 0:4, 1] = [324.0, 322.0, 162.0, 318.0]
        raw[0, 4 + 0, 1] = 0.65

        # Box 2: car (class 2) with low confidence 0.20 -> filtered out
        raw[0, 0:4, 2] = [100.0, 100.0, 80.0, 60.0]
        raw[0, 4 + 2, 2] = 0.20

        # Scale from 640x640 input space to 1280x720 original image
        detections = decode_yolo_output(
            raw,
            orig_width=1280,
            orig_height=720,
            input_width=640,
            input_height=640,
            confidence_threshold=0.5,
            iou_threshold=0.45,
        )

        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.label, "person")
        self.assertEqual(det.class_id, 0)
        self.assertAlmostEqual(det.confidence, 0.92, places=4)
        # x: (320 +- 80) * 2 = [480, 800]; y: (320 +- 160) * (720/640) = [180, 540]
        self.assertAlmostEqual(det.bbox.x1, 480.0, places=2)
        self.assertAlmostEqual(det.bbox.x2, 800.0, places=2)
        self.assertAlmostEqual(det.bbox.y1, 180.0, places=2)
        self.assertAlmostEqual(det.bbox.y2, 540.0, places=2)


if __name__ == "__main__":
    unittest.main()
