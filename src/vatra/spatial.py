"""Spatial zone geometry, affine transforms, and Physical AI ROI monitoring."""

from __future__ import annotations

import math
from typing import Sequence

from vatra.types import Detection, FramePacket, SpatialZone, ZoneEvent


def make_rect_polygon(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle_deg: float = 0.0,
) -> list[tuple[float, float]]:
    """Build a 4-corner polygon centered at (center_x, center_y) rotated by angle_deg."""
    half_w = max(1.0, float(width)) / 2.0
    half_h = max(1.0, float(height)) / 2.0
    rad = math.radians(float(angle_deg))
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    local_corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]
    return [
        (
            round(center_x + lx * cos_a - ly * sin_a, 2),
            round(center_y + lx * sin_a + ly * cos_a, 2),
        )
        for lx, ly in local_corners
    ]


def transform_polygon(
    polygon: Sequence[tuple[float, float]],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    angle_deg: float = 0.0,
) -> list[tuple[float, float]]:
    """Scale and rotate a polygon around its centroid, then translate by (dx, dy)."""
    if not polygon:
        return []

    cx = sum(pt[0] for pt in polygon) / len(polygon)
    cy = sum(pt[1] for pt in polygon) / len(polygon)
    rad = math.radians(float(angle_deg))
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    transformed: list[tuple[float, float]] = []
    for px, py in polygon:
        sx = (px - cx) * float(scale_x)
        sy = (py - cy) * float(scale_y)
        rx = sx * cos_a - sy * sin_a
        ry = sx * sin_a + sy * cos_a
        transformed.append((round(cx + rx + float(dx), 2), round(cy + ry + float(dy), 2)))
    return transformed


def point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """Return True if a 2D point lies inside a polygon via ray casting."""
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            denom = y2 - y1
            x_intersect = (x2 - x1) * (y - y1) / denom + x1
            if x < x_intersect:
                inside = not inside

    return inside


def _detection_anchor_point(det: Detection, anchor: str) -> tuple[float, float]:
    if anchor == "bottom_center":
        return (det.bbox.center[0], det.bbox.y2)
    return det.bbox.center


class SpatialZoneMonitor:
    """Evaluates detections against configured polygonal spatial zones."""

    def __init__(self, zones: list[SpatialZone]) -> None:
        self.zones = list(zones)

    def evaluate(
        self,
        packet: FramePacket,
        detections: list[Detection],
    ) -> list[ZoneEvent]:
        """Return ZoneEvents for each zone containing matching detections."""
        events: list[ZoneEvent] = []
        if not detections or not self.zones:
            return events

        for zone in self.zones:
            watched = set(zone.watched_labels)
            polygon = list(zone.polygon)
            anchor = zone.anchor
            matched: list[Detection] = []
            for det in detections:
                if watched and det.label not in watched:
                    continue
                if point_in_polygon(_detection_anchor_point(det, anchor), polygon):
                    matched.append(det)

            if matched:
                events.append(
                    ZoneEvent(
                        frame_id=packet.frame_id,
                        timestamp_ms=packet.timestamp_ms,
                        zone_id=zone.zone_id,
                        zone_name=zone.name,
                        detections=matched,
                    )
                )

        return events
