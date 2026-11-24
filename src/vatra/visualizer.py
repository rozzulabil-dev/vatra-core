"""OpenCV-backed live visualizer and video recorder for vatra-core."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vatra.types import Detection, FramePacket, SpatialZone, ZoneEvent


class FrameVisualizer:
    """Renders detections, spatial zones, and alert HUD overlays onto frames."""

    def __init__(
        self,
        zones: Sequence[SpatialZone] | None = None,
        window_name: str = "Vatra.ai Edge Vision",
        show: bool = False,
        save_path: str | None = None,
        fps: float = 30.0,
    ) -> None:
        self.zones = list(zones) if zones else []
        self.window_name = window_name
        self.show = show
        self.save_path = save_path
        self.fps = fps
        self._writer: Any = None
        self._window_opened = False

    def _ensure_writer(self, cv2: Any, width: int, height: int) -> None:
        if not self.save_path or self._writer is not None:
            return
        out_path = Path(self.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for output: {self.save_path!r}")
        self._writer = writer

    def render(
        self,
        packet: FramePacket,
        detections: Sequence[Detection],
        events: Sequence[ZoneEvent],
    ) -> tuple[Any, bool]:
        """Render frame overlays, write to video if configured, and show window."""
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python-headless and numpy are required for FrameVisualizer. "
                "Install them with: pip install 'vatra-core[vision]'"
            ) from exc

        if packet.data is not None:
            canvas = packet.data.copy()
        else:
            canvas = np.full((packet.height, packet.width, 3), 18, dtype=np.uint8)
            canvas[:, :, 0] = 28
            canvas[:, :, 1] = 22

        height, width = canvas.shape[:2]
        triggered_zones = {e.zone_id for e in events}

        # Draw spatial zones
        for zone in self.zones:
            polygon = list(zone.polygon)
            if len(polygon) < 3:
                continue
            is_alert = zone.zone_id in triggered_zones
            color = (68, 68, 239) if is_alert else (94, 197, 34)  # BGR: red or green
            pts = np.array([(int(x), int(y)) for x, y in polygon], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.16 if is_alert else 0.08, canvas, 0.84 if is_alert else 0.92, 0, canvas)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

            label_pos = (max(8, int(polygon[0][0])), max(24, int(polygon[0][1]) - 8))
            cv2.putText(
                canvas,
                zone.name,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Draw object detections
        for det in detections:
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
            box_color = (248, 189, 56)  # BGR cyan-ish
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
            caption = (
                f"{det.label} #{det.track_id} {det.confidence:.2f}"
                if det.track_id is not None
                else f"{det.label} {det.confidence:.2f}"
            )
            cv2.putText(
                canvas,
                caption,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1,
                cv2.LINE_AA,
            )

        # Draw top HUD bar
        cv2.rectangle(canvas, (0, 0), (width, 32), (15, 23, 42), -1)
        if events:
            status_text = f"ZONE ALERT: {events[0].zone_name}"
            status_color = (68, 68, 239)
        else:
            status_text = "SAFE"
            status_color = (94, 197, 34)

        hud_left = f"Vatra.ai | Frame #{packet.frame_id:04d} | Det: {len(detections)}"
        cv2.putText(
            canvas,
            hud_left,
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 244, 248),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            status_text,
            (max(width - 230, 10), 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            2,
            cv2.LINE_AA,
        )

        if self.save_path:
            self._ensure_writer(cv2, width=width, height=height)
            self._writer.write(canvas)

        keep_running = True
        if self.show:
            cv2.imshow(self.window_name, canvas)
            self._window_opened = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                keep_running = False

        return canvas, keep_running

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._window_opened:
            try:
                import cv2  # type: ignore[import-not-found]

                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self._window_opened = False
