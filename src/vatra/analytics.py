"""Per-zone occupancy, unique entry counting, and dwell-time analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from vatra.types import FramePacket, SpatialZone, ZoneEvent


@dataclass(slots=True)
class _DwellRecord:
    first_seen_ms: float
    last_seen_ms: float


@dataclass(slots=True)
class ZoneAnalyticsSnapshot:
    """Point-in-time analytics summary for a single SpatialZone."""

    zone_id: str
    zone_name: str
    active_count: int
    unique_entries: int
    avg_dwell_seconds: float
    max_dwell_seconds: float


class ZoneAnalyticsTracker:
    """Computes active occupancy, unique entries, and dwell time per zone."""

    def __init__(self, zones: Sequence[SpatialZone] | None = None) -> None:
        self._zone_names: dict[str, str] = {}
        self._active_counts: dict[str, int] = {}
        self._records: dict[str, dict[Any, _DwellRecord]] = {}

        if zones:
            for zone in zones:
                self._ensure_zone(zone.zone_id, zone.name)

    def _ensure_zone(self, zone_id: str, zone_name: str) -> None:
        self._zone_names.setdefault(zone_id, zone_name)
        self._active_counts.setdefault(zone_id, 0)
        self._records.setdefault(zone_id, {})

    def update(
        self,
        packet: FramePacket,
        events: Sequence[ZoneEvent],
    ) -> dict[str, ZoneAnalyticsSnapshot]:
        """Update zone dwell records for the current frame and return snapshots."""
        for zid in list(self._active_counts):
            self._active_counts[zid] = 0

        for event in events:
            self._ensure_zone(event.zone_id, event.zone_name)
            zone_records = self._records[event.zone_id]

            active_keys: set[Any] = set()
            for idx, det in enumerate(event.detections):
                track_key: Any = (
                    det.track_id
                    if det.track_id is not None
                    else ("untracked", packet.frame_id, idx)
                )
                active_keys.add(track_key)
                rec = zone_records.get(track_key)
                if rec is None:
                    zone_records[track_key] = _DwellRecord(
                        first_seen_ms=packet.timestamp_ms,
                        last_seen_ms=packet.timestamp_ms,
                    )
                else:
                    if packet.timestamp_ms >= rec.last_seen_ms:
                        rec.last_seen_ms = packet.timestamp_ms

            self._active_counts[event.zone_id] = len(active_keys)

        return self.summary()

    def summary(self) -> dict[str, ZoneAnalyticsSnapshot]:
        """Return current analytics snapshots keyed by zone_id."""
        snapshots: dict[str, ZoneAnalyticsSnapshot] = {}
        for zone_id, zone_name in self._zone_names.items():
            zone_records = self._records.get(zone_id, {})
            dwells = [
                max(0.0, (rec.last_seen_ms - rec.first_seen_ms) / 1000.0)
                for rec in zone_records.values()
            ]
            avg_dwell = (sum(dwells) / len(dwells)) if dwells else 0.0
            max_dwell = max(dwells) if dwells else 0.0

            snapshots[zone_id] = ZoneAnalyticsSnapshot(
                zone_id=zone_id,
                zone_name=zone_name,
                active_count=self._active_counts.get(zone_id, 0),
                unique_entries=len(zone_records),
                avg_dwell_seconds=round(avg_dwell, 3),
                max_dwell_seconds=round(max_dwell, 3),
            )
        return snapshots

    def summary_dict(self) -> dict[str, dict[str, Any]]:
        """Return JSON-serializable analytics snapshots keyed by zone_id."""
        return {zid: asdict(snap) for zid, snap in self.summary().items()}
