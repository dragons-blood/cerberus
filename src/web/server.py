"""
Local web dashboard for the Gemini Robot Dog.

Provides a browser-based interface for:
- API key configuration
- Emergency stop button
- Basic movement controls
- Natural language instruction input
- Live status (battery, connection, IMU, robot state)
- Camera feed (MJPEG stream)
- Log viewer
- Robot connectivity check (ping)

Runs on 0.0.0.0:8080 by default. Set WEB_PASSWORD in config/.env to
require a password (HTTP Basic Auth). Without a password, the dashboard
is open to anyone on the local network.
"""

import asyncio
import hmac
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request

logger = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent.parent.parent


class WebServer:
    """
    Flask-based web dashboard that runs alongside the robot.

    Shares references to the Robot's subsystems (camera, go2, gemini)
    so it can display status and send commands without duplicating connections.
    """

    def __init__(self, robot, host: str = "0.0.0.0", port: int = 8080,
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        self.robot = robot
        self.host = host
        self.port = port
        self._loop = loop  # Main asyncio event loop for scheduling coroutines
        self._thread: Optional[threading.Thread] = None
        self._app = self._create_app()

    def _create_app(self) -> Flask:
        app = Flask(
            __name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"),
        )
        app.config["SECRET_KEY"] = os.urandom(24)

        # Suppress Flask request logging in production
        flask_log = logging.getLogger("werkzeug")
        flask_log.setLevel(logging.WARNING)

        # ---- Optional HTTP Basic Auth ----
        # Set WEB_PASSWORD in config/.env to enable. Username is always "robot".
        web_password = os.environ.get("WEB_PASSWORD", "").strip()
        if web_password:
            logger.info("Web dashboard authentication ENABLED")

            @app.before_request
            def _check_auth():
                auth = request.authorization
                if not auth or not hmac.compare_digest(auth.password or "", web_password):
                    return Response(
                        "Authentication required. Set WEB_PASSWORD in config/.env.\n",
                        401,
                        {"WWW-Authenticate": 'Basic realm="Gemini Robot Dog"'},
                    )
        else:
            logger.warning(
                "Web dashboard has NO authentication. Anyone on the network can "
                "control the robot. Set WEB_PASSWORD in config/.env to secure it."
            )

        # ---- Routes ----

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/api/status")
        def api_status():
            """Return current robot status as JSON."""
            go2 = self.robot.go2
            camera = self.robot.camera
            status = go2.status

            return jsonify({
                "connected": go2.connected,
                "state": go2.state.value,
                "battery": status.battery_percent,
                "position": list(status.position),
                "velocity": list(status.velocity),
                "imu_rpy": list(status.imu_rpy),
                "camera_running": camera.is_running,
                "camera_fps": round(camera.actual_fps, 1),
                "running": self.robot._running,
            })

        @app.route("/api/ping")
        def api_ping():
            """Ping the robot to check connectivity."""
            robot_ip = self.robot.config["unitree"]["robot_ip"]
            robot_iface = self.robot.config.get("jetson", {}).get("robot_interface", "")
            try:
                ping_cmd = ["ping", "-c", "1", "-W", "2"]
                if robot_iface:
                    ping_cmd += ["-I", robot_iface]
                ping_cmd.append(robot_ip)
                result = subprocess.run(
                    ping_cmd,
                    capture_output=True, text=True, timeout=5,
                )
                ok = result.returncode == 0
                # Extract round-trip time from ping output
                rtt = None
                if ok:
                    match = re.search(r"time[=<](\d+\.?\d*)", result.stdout)
                    if match:
                        rtt = float(match.group(1))
                return jsonify({
                    "reachable": ok,
                    "ip": robot_ip,
                    "rtt_ms": rtt,
                    "detail": result.stdout.strip().split("\n")[-1] if ok else result.stderr.strip(),
                })
            except Exception as e:
                return jsonify({"reachable": False, "ip": robot_ip, "error": str(e)})

        @app.route("/api/camera/frame.jpg")
        def camera_frame():
            """Return the latest camera frame as JPEG."""
            jpeg = self.robot.camera.get_frame_jpeg(quality=80)
            if jpeg is None:
                return Response("No frame available", status=503)
            return Response(jpeg, mimetype="image/jpeg")

        @app.route("/api/camera/stream")
        def camera_stream():
            """MJPEG stream of the camera feed."""
            def generate():
                try:
                    while self.robot.camera.is_running:
                        jpeg = self.robot.camera.get_frame_jpeg(quality=70)
                        if jpeg:
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + jpeg + b"\r\n"
                            )
                        time.sleep(0.1)  # ~10 fps for the web stream
                except GeneratorExit:
                    pass  # Client disconnected — clean exit

            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/api/emergency_stop", methods=["POST"])
        def emergency_stop():
            """Trigger emergency stop — motors go limp."""
            logger.critical("EMERGENCY STOP triggered from web UI")
            self._run_async(self.robot.emergency_stop())
            return jsonify({"status": "emergency_stop_sent"})

        @app.route("/api/stop", methods=["POST"])
        def stop_movement():
            """Stop movement (robot stays standing)."""
            self._run_async(self.robot.go2.stop())
            return jsonify({"status": "ok"})

        @app.route("/api/command", methods=["POST"])
        def command():
            """Execute a movement command."""
            data = request.get_json()
            if not data or "action" not in data:
                return jsonify({"error": "Missing 'action'"}), 400

            action = data["action"]
            params = data.get("params", {})

            go2 = self.robot.go2
            actions = {
                "stand_up": go2.stand_up,
                "sit_down": go2.sit_down,
                "stop": go2.stop,
                "recovery_stand": go2.recovery_stand,
            }

            # Clamp parameters to safe bounds (same limits as guardrails)
            max_speed = self.robot.config["unitree"]["max_linear_velocity"]
            max_angular = self.robot.config["unitree"]["max_angular_velocity"]
            max_distance = 3.0  # meters per single action
            max_angle = 180.0   # degrees per single turn

            try:
                if action in actions:
                    self._run_async(actions[action]())
                    return jsonify({"status": "ok", "action": action})
                elif action == "move_forward":
                    distance = max(0.1, min(float(params.get("distance", 0.5)), max_distance))
                    speed = max(0.1, min(float(params.get("speed", 0.3)), max_speed))
                    self._run_async(go2.move_forward(distance=distance, speed=speed))
                    return jsonify({"status": "ok", "action": action})
                elif action == "move_backward":
                    distance = max(0.1, min(float(params.get("distance", 0.5)), max_distance))
                    speed = max(0.1, min(float(params.get("speed", 0.3)), max_speed))
                    self._run_async(go2.move_backward(distance=distance, speed=speed))
                    return jsonify({"status": "ok", "action": action})
                elif action == "turn_left":
                    angle = max(1.0, min(float(params.get("angle", 30)), max_angle))
                    self._run_async(go2.turn(angle_degrees=angle))
                    return jsonify({"status": "ok", "action": action})
                elif action == "turn_right":
                    angle = max(1.0, min(float(params.get("angle", 30)), max_angle))
                    self._run_async(go2.turn(angle_degrees=-angle))
                    return jsonify({"status": "ok", "action": action})
                else:
                    return jsonify({"error": f"Unknown action: {action}"}), 400
            except (ValueError, TypeError) as e:
                return jsonify({"error": f"Invalid parameters: {e}"}), 400

        @app.route("/api/instruction", methods=["POST"])
        def instruction():
            """Send a natural language instruction to Gemini."""
            data = request.get_json()
            if not data or "instruction" not in data:
                return jsonify({"error": "Missing 'instruction'"}), 400

            text = data["instruction"].strip()
            if not text:
                return jsonify({"error": "Empty instruction"}), 400
            if len(text) > 1000:
                return jsonify({"error": "Instruction too long (max 1000 chars)"}), 400

            # Run async in background so the HTTP request returns immediately
            def _run():
                self._run_async(self.robot.run_instruction(text))

            threading.Thread(target=_run, daemon=True).start()
            return jsonify({"status": "instruction_sent", "instruction": text})

        @app.route("/api/key", methods=["GET"])
        def get_key_status():
            """Check if API key is configured (does not reveal the key)."""
            api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            has_key = bool(api_key) and api_key != "your_gemini_api_key_here"
            return jsonify({
                "configured": has_key,
                "key_preview": f"{api_key[:8]}...{api_key[-4:]}" if has_key and len(api_key) > 12 else None,
            })

        @app.route("/api/key", methods=["POST"])
        def set_key():
            """Set the Gemini API key (writes to config/.env)."""
            data = request.get_json()
            if not data or "key" not in data:
                return jsonify({"error": "Missing 'key'"}), 400

            new_key = data["key"].strip()
            if not new_key:
                return jsonify({"error": "Empty key"}), 400

            # Basic format validation
            if not new_key.startswith("AIza"):
                return jsonify({"error": "Invalid key format — Gemini keys start with 'AIza'"}), 400

            # Sanitize: API keys are alphanumeric + hyphens/underscores only.
            # Reject newlines or special chars that could inject into .env.
            if not re.match(r'^[A-Za-z0-9_\-]+$', new_key):
                return jsonify({"error": "Invalid characters in key"}), 400

            env_file = REPO_DIR / "config" / ".env"

            # Read existing .env or start fresh
            if env_file.exists():
                content = env_file.read_text()
                # Replace existing key
                if "GOOGLE_API_KEY=" in content:
                    content = re.sub(
                        r"GOOGLE_API_KEY=.*",
                        f"GOOGLE_API_KEY={new_key}",
                        content,
                    )
                else:
                    content += f"\nGOOGLE_API_KEY={new_key}\n"
            else:
                content = f"GOOGLE_API_KEY={new_key}\n"

            env_file.write_text(content)

            # Update the running process's environment
            os.environ["GOOGLE_API_KEY"] = new_key

            # Reinitialize the Gemini client with the new key
            try:
                from src.gemini.robotics_client import GeminiRoboticsClient
                self.robot.gemini = GeminiRoboticsClient(
                    model_id=self.robot.config["gemini"]["model_id"],
                    thinking_budget=self.robot.config["gemini"]["thinking_budget"],
                    max_output_tokens=self.robot.config["gemini"]["max_output_tokens"],
                )
                logger.info("API key updated and Gemini client reinitialized via web UI")
            except Exception as e:
                logger.warning("API key saved but Gemini client failed to reinitialize: %s", e)

            return jsonify({"status": "ok", "key_preview": f"{new_key[:8]}...{new_key[-4:]}"})

        @app.route("/api/logs")
        def get_logs():
            """Return the last N lines of the robot log."""
            try:
                n = int(request.args.get("lines", 50))
            except (ValueError, TypeError):
                n = 50
            n = max(1, min(n, 500))  # Clamp 1-500
            log_file = REPO_DIR / "logs" / "robot.log"
            if not log_file.exists():
                return jsonify({"lines": [], "count": 0})
            try:
                # Read only the tail to avoid loading huge files into memory
                # Read last ~100KB (generous for 500 lines)
                max_bytes = 100_000
                size = log_file.stat().st_size
                with open(log_file, "r") as f:
                    if size > max_bytes:
                        f.seek(size - max_bytes)
                        f.readline()  # Skip partial first line
                    lines = f.readlines()
                tail = [l.rstrip("\n") for l in lines[-n:]]
                return jsonify({"lines": tail, "count": len(tail)})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return app

    def _run_async(self, coro):
        """Run an async coroutine from a sync Flask context."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            asyncio.run(coro)

    def start(self):
        """Start the web server in a background thread."""
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info("Web dashboard started at http://%s:%d", self.host, self.port)

    def _run_server(self):
        """Run Flask in a background thread."""
        self._app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    def stop(self):
        """Stop the web server (best-effort — Flask doesn't have clean shutdown)."""
        logger.info("Web dashboard stopping")
