"""Zero-dependency local web dashboard and MJPEG stream server for vatra-core."""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Sequence

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
        max-width: 1120px;
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
        grid-template-columns: 1.3fr 1fr;
        gap: 1rem;
        align-items: start;
      }
      .stack {
        display: grid;
        gap: 1rem;
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
      .card-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
      }
      .card-head h2 {
        margin: 0;
      }
      .btn-reset {
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        border: 1px solid var(--border);
        background: #f1f5f9;
        color: var(--text);
        font-size: 0.78rem;
        font-weight: 600;
        cursor: pointer;
      }
      .btn-reset:hover {
        background: #e2e8f0;
      }
      .controls-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem 1rem;
      }
      .ctrl {
        display: grid;
        gap: 0.25rem;
      }
      .ctrl-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: var(--muted);
        font-weight: 600;
      }
      .ctrl-label span:last-child {
        color: var(--text);
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      }
      .ctrl input[type="range"],
      .ctrl select {
        width: 100%;
        accent-color: var(--blue);
      }
      .ctrl select {
        padding: 0.32rem 0.45rem;
        border-radius: 6px;
        border: 1px solid var(--border);
        background: #ffffff;
        color: var(--text);
        font-size: 0.84rem;
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
        max-height: 240px;
        overflow-y: auto;
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        font-size: 0.8rem;
      }
      .events-list li {
        padding: 0.42rem 0;
        border-bottom: 1px solid var(--border);
      }
      @media (max-width: 860px) {
        .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .main-grid { grid-template-columns: 1fr; }
      }
      @media (max-width: 520px) {
        .controls-grid { grid-template-columns: 1fr; }
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
        <div class="stack">
          <div class="card stream-box">
            <h2>Annotated Stream</h2>
            <img src="/stream.mjpg" alt="Vatra.ai Live Video Stream" />
          </div>

          <div class="card" id="zone-controls-card">
            <div class="card-head">
              <h2>Zone &amp; Accuracy Controls</h2>
              <button class="btn-reset" type="button" id="ctrl-reset">Reset</button>
            </div>
            <div class="controls-grid">
              <div class="ctrl">
                <div class="ctrl-label"><span>Position X</span><span id="val-x">320 px</span></div>
                <input id="ctrl-x" type="range" min="0" max="640" value="320" />
              </div>
              <div class="ctrl">
                <div class="ctrl-label"><span>Position Y</span><span id="val-y">240 px</span></div>
                <input id="ctrl-y" type="range" min="0" max="480" value="240" />
              </div>
              <div class="ctrl">
                <div class="ctrl-label"><span>Scale Width</span><span id="val-w">320 px</span></div>
                <input id="ctrl-w" type="range" min="40" max="640" value="320" />
              </div>
              <div class="ctrl">
                <div class="ctrl-label"><span>Scale Height</span><span id="val-h">240 px</span></div>
                <input id="ctrl-h" type="range" min="40" max="480" value="240" />
              </div>
              <div class="ctrl">
                <div class="ctrl-label"><span>Rotation</span><span id="val-rot">0°</span></div>
                <input id="ctrl-rot" type="range" min="-180" max="180" value="0" />
              </div>
              <div class="ctrl">
                <div class="ctrl-label"><span>Confidence</span><span id="val-conf">0.50</span></div>
                <input id="ctrl-conf" type="range" min="0.20" max="0.90" step="0.05" value="0.50" />
              </div>
              <div class="ctrl">
                <div class="ctrl-label"><span>Zone Anchor</span><span>point</span></div>
                <select id="ctrl-anchor">
                  <option value="center">Center (box midpoint)</option>
                  <option value="bottom_center">Bottom Center (feet/base)</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <div class="stack">
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
      let controlsInitialized = false;
      let lastFrameDims = "";
      let pendingTimer = null;

      function applyControlValues(zc) {
        if (!zc || zc.width == null) return;
        const fw = Math.max(160, Math.round(zc.frame_width || 640));
        const fh = Math.max(120, Math.round(zc.frame_height || 480));
        const dimsKey = `${fw}x${fh}`;
        const xEl = document.getElementById("ctrl-x");
        const yEl = document.getElementById("ctrl-y");
        const wEl = document.getElementById("ctrl-w");
        const hEl = document.getElementById("ctrl-h");
        const rotEl = document.getElementById("ctrl-rot");
        const confEl = document.getElementById("ctrl-conf");
        const anchorEl = document.getElementById("ctrl-anchor");

        if (dimsKey !== lastFrameDims) {
          xEl.max = fw;
          wEl.max = fw;
          yEl.max = fh;
          hEl.max = fh;
          lastFrameDims = dimsKey;
        }

        xEl.value = Math.round(zc.center_x);
        yEl.value = Math.round(zc.center_y);
        wEl.value = Math.round(zc.width);
        hEl.value = Math.round(zc.height);
        rotEl.value = Math.round(zc.angle_deg || 0);
        confEl.value = Number(zc.confidence ?? 0.5).toFixed(2);
        anchorEl.value = zc.anchor || "center";
        updateControlLabels();
      }

      function updateControlLabels() {
        document.getElementById("val-x").textContent = document.getElementById("ctrl-x").value + " px";
        document.getElementById("val-y").textContent = document.getElementById("ctrl-y").value + " px";
        document.getElementById("val-w").textContent = document.getElementById("ctrl-w").value + " px";
        document.getElementById("val-h").textContent = document.getElementById("ctrl-h").value + " px";
        document.getElementById("val-rot").textContent = document.getElementById("ctrl-rot").value + "°";
        document.getElementById("val-conf").textContent = Number(document.getElementById("ctrl-conf").value).toFixed(2);
      }

      async function sendZoneUpdate(payload) {
        try {
          const res = await fetch("/api/zone", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (res.ok && payload.reset) {
            const updated = await res.json();
            applyControlValues(updated);
          }
        } catch (err) {
          console.debug(err);
        }
      }

      function scheduleZonePush() {
        updateControlLabels();
        if (pendingTimer) clearTimeout(pendingTimer);
        pendingTimer = setTimeout(() => {
          sendZoneUpdate({
            center_x: Number(document.getElementById("ctrl-x").value),
            center_y: Number(document.getElementById("ctrl-y").value),
            width: Number(document.getElementById("ctrl-w").value),
            height: Number(document.getElementById("ctrl-h").value),
            angle_deg: Number(document.getElementById("ctrl-rot").value),
            confidence: Number(document.getElementById("ctrl-conf").value),
            anchor: document.getElementById("ctrl-anchor").value,
          });
        }, 40);
      }

      ["ctrl-x", "ctrl-y", "ctrl-w", "ctrl-h", "ctrl-rot", "ctrl-conf", "ctrl-anchor"].forEach((id) => {
        const el = document.getElementById(id);
        el.addEventListener("input", scheduleZonePush);
        el.addEventListener("change", scheduleZonePush);
      });
      document.getElementById("ctrl-reset").addEventListener("click", () => {
        sendZoneUpdate({ reset: true });
      });

      async function refreshStatus() {
        try {
          const res = await fetch("/api/status");
          if (!res.ok) return;
          const data = await res.json();
          const st = data.stats || {};
          document.getElementById("meta-line").textContent = "vatra-core v" + (data.version || "0.6.0");
          document.getElementById("stat-frames").textContent = st.frames_processed ?? 0;
          document.getElementById("stat-fps").textContent = (st.avg_fps ?? 0).toFixed(1);
          document.getElementById("stat-dets").textContent = st.total_detections ?? 0;
          document.getElementById("stat-events").textContent = st.total_zone_events ?? 0;

          if (data.zone_control) {
            const fw = Math.max(160, Math.round(data.zone_control.frame_width || 640));
            const fh = Math.max(120, Math.round(data.zone_control.frame_height || 480));
            if (!controlsInitialized || `${fw}x${fh}` !== lastFrameDims) {
              applyControlValues(data.zone_control);
              controlsInitialized = true;
            }
          }

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

    def __init__(self, version: str = "0.6.0", max_events: int = 25) -> None:
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
        self._zone_control: dict[str, Any] = {}

    def update(
        self,
        *,
        jpeg_bytes: bytes | None = None,
        stats: dict[str, Any] | None = None,
        analytics: dict[str, dict[str, Any]] | None = None,
        events: Sequence[ZoneEvent] | None = None,
        zone_control: dict[str, Any] | None = None,
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
            if zone_control is not None:
                self._zone_control = dict(zone_control)
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
                "zone_control": dict(self._zone_control),
            }


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _make_handler(
    state: DashboardState,
    running_event: threading.Event,
    on_zone_update: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> type[BaseHTTPRequestHandler]:
    class VatraRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
                self._send_json(200, state.snapshot())
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

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path != "/api/zone":
                self.send_error(404, "Not Found")
                return

            if on_zone_update is None:
                self._send_json(501, {"error": "Zone controls are not configured."})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object.")
                updated = on_zone_update(payload)
                state.update(zone_control=updated)
                self._send_json(200, updated)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})

    return VatraRequestHandler


class VatraDashboardServer:
    """Runs the Vatra.ai HTTP dashboard and MJPEG video feed server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        state: DashboardState | None = None,
        on_zone_update: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.host = host
        self.state = state or DashboardState()
        self._running = threading.Event()
        self._running.set()
        handler_cls = _make_handler(
            self.state,
            self._running,
            on_zone_update=on_zone_update,
        )
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
