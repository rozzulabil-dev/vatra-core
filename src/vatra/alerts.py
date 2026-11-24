"""Structured alert dispatchers (JSONL logging and rate-limited HTTP webhooks)."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from vatra.types import ZoneEvent


def zone_event_to_dict(event: ZoneEvent) -> dict[str, Any]:
    """Serialize a ZoneEvent into a JSON-compatible dictionary."""
    return {
        "frame_id": event.frame_id,
        "timestamp_ms": event.timestamp_ms,
        "zone_id": event.zone_id,
        "zone_name": event.zone_name,
        "detections": [
            {
                "label": det.label,
                "confidence": round(float(det.confidence), 4),
                "class_id": det.class_id,
                "track_id": det.track_id,
                "bbox": {
                    "x1": det.bbox.x1,
                    "y1": det.bbox.y1,
                    "x2": det.bbox.x2,
                    "y2": det.bbox.y2,
                },
            }
            for det in event.detections
        ],
    }


def _default_http_sender(url: str, payload: bytes, timeout: float) -> None:
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout):
        pass


class JsonlAlertSink:
    """Appends one JSON line per ZoneEvent to a local JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def emit(self, event: ZoneEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(zone_event_to_dict(event)) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)


class WebhookAlertSink:
    """Dispatches rate-limited JSON alerts to an HTTP webhook endpoint."""

    def __init__(
        self,
        url: str,
        cooldown_seconds: float = 5.0,
        timeout_seconds: float = 3.0,
        sender: Callable[[str, bytes, float], None] | None = None,
    ) -> None:
        if cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be >= 0.0.")
        self.url = url
        self.cooldown_seconds = cooldown_seconds
        self.timeout_seconds = timeout_seconds
        self._sender = sender or _default_http_sender
        self._last_sent: dict[str, float] = {}

    def emit(self, event: ZoneEvent) -> bool:
        """Send webhook payload unless suppressed by per-zone cooldown."""
        now = time.monotonic()
        last = self._last_sent.get(event.zone_id)
        if last is not None and (now - last) < self.cooldown_seconds:
            return False

        payload = json.dumps(zone_event_to_dict(event)).encode("utf-8")
        self._sender(self.url, payload, self.timeout_seconds)
        self._last_sent[event.zone_id] = now
        return True
