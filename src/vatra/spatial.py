"""Spatial zone geometry and Physical AI region-of-interest monitoring."""

from __future__ import annotations

from vatra.types import Detection, FramePacket, SpatialZone, ZoneEvent


def point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
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
            matched: list[Detection] = []
            for det in detections:
                if watched and det.label not in watched:
                    continue
                if point_in_polygon(det.bbox.center, zone.polygon):
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
