"""
Live 3D Web Visualization Server

Flask-SocketIO server that serves a Three.js 3D scene and receives
real-time position/hazard/trajectory data from the VESTA pipeline
via WebSocket push.

Usage:
    Launched automatically by `realtime/run.py --viz`
    Opens browser to http://localhost:8080
"""

import threading
from pathlib import Path

from flask import Flask, send_from_directory
from flask_socketio import SocketIO


class LiveVizServer:
    """
    WebSocket server that pushes pipeline state to a Three.js 3D scene.

    The pipeline calls `push_state(data)` every few frames.
    The server emits 'state_update' events to all connected browsers.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None

        static_dir = Path(__file__).parent / "static"

        self.app = Flask(__name__, static_folder=str(static_dir))
        self.app.config["SECRET_KEY"] = "vesta-viz"
        self.sio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")

        # Serve the main page
        @self.app.route("/")
        def index():
            return send_from_directory(str(static_dir), "index.html")

        @self.sio.on("connect")
        def on_connect():
            print("[VESTA VIZ] Browser connected")

        @self.sio.on("disconnect")
        def on_disconnect():
            print("[VESTA VIZ] Browser disconnected")

    def start(self):
        """Start the server in a background daemon thread."""
        def _run():
            self.sio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                log_output=False,
            )

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        print(f"[VESTA VIZ] 3D visualization at http://localhost:{self.port}")

    def push_state(self, data: dict):
        """
        Push pipeline state to all connected browsers.

        Expected data format:
        {
            "worker": {"x": float, "y": float, "heading": float},
            "path": [[x, y], ...],
            "hazards": [{"x": float, "y": float, "label": str, "severity": str, ...}],
            "prediction": {
                "points": [[x, y], ...],
                "collisions": [{"hazard_id": str, "label": str, "eta_seconds": float}]
            }
        }
        """
        self.sio.emit("state_update", data)

    def shutdown(self):
        """Stop the server (best-effort, it's a daemon thread)."""
        pass  # Daemon thread dies with main process
