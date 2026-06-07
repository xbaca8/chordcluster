"""
Web server for the chord-transition KNN similarity graph.

Serves the static page in `knn_graph/` and exposes a single API endpoint that
adds an uploaded song to one of the graphs:

    POST /api/add_song
        body    : raw audio file bytes
        headers : X-Song-Title, X-Song-Artist, X-Song-Filename (URL-encoded)
                  X-Song-Duration (integer seconds, 1–300, default 30)
        returns : JSON {mode, file_name, title, artist, key, romanized_chords,
                        n_nodes}

Raw bytes + headers are used instead of multipart/form-data because Python 3.13
removed the `cgi` module. One server hosts both the page and the API, so the
browser talks to the same origin (no CORS).

Run:
    uv run python code/server.py        # then open http://localhost:8080
"""

from __future__ import annotations

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import graph_pipeline as gp

PORT = 8080
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB cap

# analyze_song mutates shared tables/JSON; serialize uploads.
_add_lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    # Serve static files out of the knn_graph/ directory.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(gp.GRAPH_DIR), **kwargs)

    # Quieter, single-line logging.
    def log_message(self, fmt, *args):
        print(f"[server] {self.address_string()} - {fmt % args}")

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Default: redirect bare root to the page.
        if self.path in ("/", ""):
            self.path = "/knn_graph.html"
        return super().do_GET()

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/add_song":
            self._handle_add_song()
        elif path == "/api/delete_song":
            self._handle_delete_song()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_add_song(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0

        if length <= 0:
            self._send_json(400, {"error": "Empty upload."})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"error": "File too large (max 100 MB)."})
            return

        title    = unquote(self.headers.get("X-Song-Title", ""))
        artist   = unquote(self.headers.get("X-Song-Artist", ""))
        filename = unquote(self.headers.get("X-Song-Filename", ""))
        try:
            duration = max(1, min(300, int(self.headers.get("X-Song-Duration", "30"))))
        except ValueError:
            duration = 30
        audio    = self.rfile.read(length)

        try:
            with _add_lock:
                result = gp.add_song(audio, title, artist, filename, duration=duration)
            self._send_json(200, result)
        except ValueError as e:
            # Expected, user-facing failure (e.g. no usable transitions).
            self._send_json(422, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - report anything else as 500
            import traceback
            traceback.print_exc()
            self._send_json(500, {"error": f"Analysis failed: {e}"})

    def _handle_delete_song(self):
        try:
            payload = json.loads(self._read_body() or b"{}")
        except (json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        file_name = (payload.get("file_name") or "").strip()
        mode = payload.get("mode")
        if not file_name:
            self._send_json(400, {"error": "Missing 'file_name'."})
            return

        try:
            with _add_lock:
                result = gp.delete_song(file_name, mode)
            self._send_json(200, result)
        except ValueError as e:
            self._send_json(422, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._send_json(500, {"error": f"Delete failed: {e}"})


def main():
    # Make sure the tables + graph JSON exist before we start serving.
    print("Preparing graph data…")
    counts = gp.rebuild_all_graphs()
    print(f"Loaded graphs: {counts}")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving on http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
