"""Zero-dependency local web dashboard and MJPEG stream server for vatra-core."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence

from vatra.alerts import zone_event_to_dict
from vatra.types import ZoneEvent

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vatra.ai — Live Edge Vision Dashboard</title>
    <style>
      :root {
        --bg: #f8fafc;
        --card: #ffffff;
        --border: #e2e8f0;
        --text: #0f172a;
        --muted: #64748b;
        --blue: #2563eb;
        --green: #16a34a;
        --red: #dc2626;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.5;
      }
      .wrap {
        max-width: 1080px;
        margin: 0 auto;
        padding: 1.25rem;
      }
      header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 1.25rem;
      }
      h1 {
        margin: 0;
        font-size: 1.35rem;
      }
      .pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #dcfce7;
        color: var(--green);
      }
      .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.25rem;
      }
      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
      }
      .stat-label {
        font-size: 0.78rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .stat-val {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.15rem;
      }
      .main-grid {
        display: grid;
        grid-template-columns: 1.35fr 1fr;
        gap: 1rem;
        align-items: start;
      }
      .stream-box img {
        display: block;
        width: 100%;
        height: auto;
        border-radius: 8px;
        background: #0f172a;
      }
      h2 {
        margin: 0 0 0.75rem;
        font-size: 1.02rem;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
      }
      th, td {
        text-align: left;
        padding: 0.5rem 0.4rem;
        border-bottom: 1px solid var(--border);
      }
      th { color: var(--muted); font-weight: 600; }
      .events-list {
        list-style: none;
        margin: 0;
        padding: 0;
        max-height: 290px;
        overflow-y: auto;
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        font-size: 0.8rem;
      }
      .events-list li {
        padding: 0.42rem 0;
        border-bottom: 1px solid var(--border);
      }
      @media (max-width: 820px) {
        .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .main-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <div>
          <h1>Vatra.ai Edge Vision Dashboard</h1>
          <small id="meta-line" style="color: var(--muted);">vatra-core live telemetry</small>
        </div>
        <span class="pill" id="live-badge">LIVE</span>
      </header>

      <div class="stats-grid">
        <div class="card">
          <div class="stat-label">Frames</div>
          <div class="stat-val" id="stat-frames">0</div>
        </div>
        <div class="card">
          <div class="stat-label">Avg FPS</div>
          <div class="stat-val" id="stat-fps">0.0</div>
        </div>
        <div class="card">
          <div class="stat-label">Detections</div>
          <div class="stat-val" id="stat-dets">0</div>
        </div>
        <div class="card">
          <div class="stat-label">Zone Events</div>
          <div class="stat-val" id="stat-events">0</div>
        </div>
      </div>

      <div class="main-grid">
        <div class="card stream-box">
          <h2>Annotated Stream</h2>
          <img src="/stream.mjpg" alt="Vatra.ai Live Video Stream" />
        </div>

        <div style="display: grid; gap: 1rem;">
          <div class="card">
            <h2>Zone Analytics</h2>
            <table>
              <thead>
                <tr>
                  <th>Zone</th>
                  <th>Active</th>
                  <th>Unique</th>
                  <th>Avg Dwell</th>
                  <th>Max Dwell</th>
                </tr>
              </thead>
              <tbody id="analytics-body">
                <tr><td colspan="5" style="color: var(--muted);">Waiting for frames...</td></tr>
              </tbody>
            </table>
          </div>

          <div class="card">
            <h2>Recent Zone Events</h2>
            <ul class="events-list" id="events-list">
              <li style="color: var(--muted);">No zone events yet.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <script>
      async function refreshStatus() {
        try {
          const res = await fetch("/api/status");
          if (!res.ok) return;
          const data = await res.json();
          const st = data.stats || {};
          document.getElementById("meta-line").textContent = "vatra-core v" + (data.version || "0.4.0");
          document.getElementById("stat-frames").textContent = st.frames_processed ?? 0;
          document.getElementById("stat-fps").textContent = (st.avg_fps ?? 0).toFixed(1);
          document.getElementById("stat-dets").textContent = st.total_detections ?? 0;
          document.getElementById("stat-events").textContent = st.total_zone_events ?? 0;

          const analytics = Object.values(data.analytics || {});
          const tbody = document.getElementById("analytics-body");
          if (analytics.length > 0) {
            tbody.innerHTML = analytics.map((z) =>
              `<tr>
                <td><strong>${z.zone_name}</strong></td>
                <td>${z.active_count}</td>
                <td>${z.unique_entries}</td>
                <td>${Number(z.avg_dwell_seconds).toFixed(2)}s</td>
                <td>${Number(z.max_dwell_seconds).toFixed(2)}s</td>
              </tr>`
            ).join("");
          }

          const events = data.recent_events || [];
          const list = document.getElementById("events-list");
          if (events.length > 0) {
            list.innerHTML = events.slice().reverse().map((ev) => {
              const tags = (ev.detections || []).map((d) =>
                d.track_id != null ? `${d.label}#${d.track_id}` : d.label
              ).join(", ");
              return `<li>[frame ${ev.frame_id}] ${ev.zone_name}: ${tags}</li>`;
            }).join("");
          }
        } catch (err) {
          console.debug(err);
        }
      }
      refreshStatus();
      setInterval(refreshStatus, 500);
    </script>
  </body>
</html>
"""


class DashboardState:
    """Thread-safe shared state exposed by VatraDashboardServer."""

    def __init__(self, version: str = "0.4.0", max_events: int = 25) -> None:
        self.version = version
        self._cond = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._frame_seq = 0
        self._stats: dict[str, Any] = {
            "frames_processed": 0,
            "total_detections": 0,
            "total_zone_events": 0,
            "avg_fps": 0.0,
        }
        self._analytics: dict[str, dict[str, Any]] = {}
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=max_events)

    def update(
        self,
        *,
        jpeg_bytes: bytes | None = None,
        stats: dict[str, Any] | None = None,
        analytics: dict[str, dict[str, Any]] | None = None,
        events: Sequence[ZoneEvent] | None = None,
    ) -> None:
        with self._cond:
            if jpeg_bytes is not None:
                self._latest_jpeg = jpeg_bytes
                self._frame_seq += 1
            if stats is not None:
                self._stats = dict(stats)
            if analytics is not None:
                self._analytics = {k: dict(v) for k, v in analytics.items()}
            if events:
                for event in events:
                    self._recent_events.append(zone_event_to_dict(event))
            self._cond.notify_all()

    def notify_all(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def wait_for_frame(
        self,
        last_seq: int,
        timeout: float = 0.5,
    ) -> tuple[bytes | None, int]:
        with self._cond:
            if self._frame_seq == last_seq:
                self._cond.wait(timeout=timeout)
            return self._latest_jpeg, self._frame_seq

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {
                "version": self.version,
                "stats": dict(self._stats),
                "analytics": {k: dict(v) for k, v in self._analytics.items()},
                "recent_events": list(self._recent_events),
            }


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _make_handler(
    state: DashboardState,
    running_event: threading.Event,
) -> type[BaseHTTPRequestHandler]:
    class VatraRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                body = _DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/status":
                body = json.dumps(state.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/stream.mjpg":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

                last_seq = -1
                while running_event.is_set():
                    jpeg_bytes, seq = state.wait_for_frame(last_seq, timeout=0.4)
                    if not running_event.is_set():
                        break
                    if jpeg_bytes is None or seq == last_seq:
                        continue
                    last_seq = seq
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii")
                    )
                    try:
                        self.wfile.write(header + jpeg_bytes + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
                return

            self.send_error(404, "Not Found")

    return VatraRequestHandler


class VatraDashboardServer:
    """Runs the Vatra.ai HTTP dashboard and MJPEG video feed server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        state: DashboardState | None = None,
    ) -> None:
        self.host = host
        self.state = state or DashboardState()
        self._running = threading.Event()
        self._running.set()
        handler_cls = _make_handler(self.state, self._running)
        self._httpd = _ReusableThreadingHTTPServer((host, port), handler_cls)
        self._thread: threading.Thread | None = None

    @property
    def server_address(self) -> tuple[str, int]:
        addr = self._httpd.server_address
        return (str(addr[0]), int(addr[1]))

    @property
    def port(self) -> int:
        return self.server_address[1]

    @property
    def url(self) -> str:
        host, port = self.server_address
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        return f"http://{display_host}:{port}"

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="vatra-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self.state.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
