"""Lightweight IoU-based multi-object tracker for vatra-core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from vatra.types import BoundingBox, Detection
from vatra.yolo import compute_iou


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: BoundingBox
    class_id: int
    label: str
    missed: int = 0
    hits: int = 1


class IouTracker:
    """Zero-dependency greedy IoU tracker assigning persistent track IDs across frames."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0.")
        if max_missed < 0:
            raise ValueError("max_missed must be >= 0.")
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    @property
    def active_tracks_count(self) -> int:
        """Number of currently retained tracks."""
        return len(self._tracks)

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        """Match detections to active tracks and return detections annotated with track_id."""
        if not detections:
            to_remove: list[int] = []
            for tid, state in self._tracks.items():
                state.missed += 1
                if state.missed > self.max_missed:
                    to_remove.append(tid)
            for tid in to_remove:
                del self._tracks[tid]
            return []

        # Build candidate (iou, track_id, det_idx) pairs for same-class matches
        pairs: list[tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for tid, state in self._tracks.items():
                if state.class_id != det.class_id:
                    continue
                iou = compute_iou(state.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, tid, det_idx))

        # Greedy highest-IoU one-to-one matching
        pairs.sort(key=lambda item: item[0], reverse=True)
        matched_tracks: set[int] = set()
        det_to_track: dict[int, int] = {}

        for _, tid, det_idx in pairs:
            if tid in matched_tracks or det_idx in det_to_track:
                continue
            matched_tracks.add(tid)
            det_to_track[det_idx] = tid
            state = self._tracks[tid]
            det = detections[det_idx]
            state.bbox = det.bbox
            state.label = det.label
            state.missed = 0
            state.hits += 1

        # Age unmatched tracks and prune expired ones
        expired: list[int] = []
        for tid, state in self._tracks.items():
            if tid not in matched_tracks:
                state.missed += 1
                if state.missed > self.max_missed:
                    expired.append(tid)
        for tid in expired:
            del self._tracks[tid]

        # Spawn new tracks for unmatched detections and build output list
        tracked_detections: list[Detection] = []
        for det_idx, det in enumerate(detections):
            tid = det_to_track.get(det_idx)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    bbox=det.bbox,
                    class_id=det.class_id,
                    label=det.label,
                )
            tracked_detections.append(replace(det, track_id=tid))

        return tracked_detections
