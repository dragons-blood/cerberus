"""
Unitree Go2 controller using WebRTC.

This module provides high-level control of the Unitree Go2 robot dog.
Uses the community go2-webrtc-connect library (from PyPI) for the WebRTC
transport layer, which handles signaling, data channel, and heartbeat.

Our controller adds on top:
  - High-level movement commands (move_forward, turn, etc.)
  - Safety features (interruptible sleep, connection monitoring, velocity clamps)
  - Emergency stop
  - State tracking

Compatibility:
  - Go2 AIR / PRO / PRO 2 / EDU (all via WebRTC, no firmware changes)

Install the transport library:
  pip install go2-webrtc-connect

Reference: https://github.com/legion1581/go2_webrtc_connect
"""

import asyncio
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RobotState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    STANDING = "standing"
    SITTING = "sitting"
    MOVING = "moving"
    DAMPED = "damped"


@dataclass
class Go2Status:
    """Current robot status from state feedback."""
    state: RobotState
    battery_percent: float
    position: tuple[float, float, float]  # x, y, z
    velocity: tuple[float, float, float]  # vx, vy, vyaw
    imu_rpy: tuple[float, float, float]   # roll, pitch, yaw (radians)
    timestamp: float


# WebRTC Sport Mode API IDs (from Unitree's webrtc_bridge protocol)
class SportAPI:
    """Sport mode API command IDs for the Go2."""
    STAND_UP = 1001
    STAND_DOWN = 1002  # Lie down / sit
    BALANCE_STAND = 1003
    STOP_MOVE = 1004
    MOVE = 1008        # Velocity move (vx, vy, vyaw)
    DAMP = 1005        # Emergency soft stop
    RECOVERY_STAND = 1006
    SWITCH_GAIT = 1011
    HEART = 1012       # Heartbeat keepalive


class Go2Controller:
    """
    High-level controller for the Unitree Go2 robot dog via WebRTC.

    Uses go2-webrtc-connect (community library) for the WebRTC transport,
    with a fallback to raw aiortc if the library isn't available.

    Usage:
        controller = Go2Controller(robot_ip="192.168.12.1")
        await controller.connect()
        await controller.stand_up()
        await controller.move(vx=0.3, vy=0.0, vyaw=0.0)
        await controller.stop()
        await controller.disconnect()
    """

    def __init__(self, robot_ip: str = "192.168.12.1",
                 max_linear_vel: float = 0.8,
                 max_angular_vel: float = 1.0,
                 token: str = "",
                 connection_method: str = "ap"):
        self.robot_ip = robot_ip
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.token = token
        self.connection_method = connection_method  # "ap" or "sta"
        self.state = RobotState.DISCONNECTED
        self._connected_event = threading.Event()  # Thread-safe connection flag
        self._on_disconnect_callback = None  # Set by Robot orchestrator

        # Transport layer — set during connect()
        self._conn = None         # go2-webrtc-connect connection object
        self._pc = None           # RTCPeerConnection (fallback)
        self._dc = None           # RTCDataChannel (fallback)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None
        self._camera = None       # Camera instance for pushing WebRTC video frames
        self._use_community_lib = False

        self._status = Go2Status(
            state=RobotState.DISCONNECTED,
            battery_percent=0.0,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            imu_rpy=(0.0, 0.0, 0.0),
            timestamp=0.0,
        )

    @property
    def connected(self) -> bool:
        """Thread-safe check of connection status."""
        return self._connected_event.is_set()

    def set_camera(self, camera):
        """
        Set a Camera instance to receive WebRTC video frames.

        When connected, frames from the Go2's video track are decoded and
        pushed to camera.push_frame(). This is how the camera works in
        STA-L mode, where UDP multicast doesn't cross the WiFi router.
        """
        self._camera = camera

    async def connect(self):
        """
        Establish WebRTC connection to the Go2.

        Tries go2-webrtc-connect library first (community standard), then
        falls back to raw aiortc if the library isn't installed.
        """
        self.state = RobotState.CONNECTING

        # Try community library first
        try:
            await self._connect_community_lib()
            self._use_community_lib = True
            return
        except ImportError:
            logger.info("go2-webrtc-connect not installed, using raw aiortc fallback")
        except Exception as e:
            logger.warning("go2-webrtc-connect failed (%s), trying aiortc fallback", e)

        # Fallback to raw aiortc
        await self._connect_aiortc()

    async def _connect_community_lib(self):
        """Connect using the go2-webrtc-connect community library."""
        from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection
        from go2_webrtc_driver.constants import WebRTCConnectionMethod

        # Map config string to WebRTCConnectionMethod enum
        method_map = {
            "ap": WebRTCConnectionMethod.LocalAP,
            "sta": WebRTCConnectionMethod.LocalSTA,
            "remote": WebRTCConnectionMethod.Remote,
        }
        method = method_map.get(self.connection_method, WebRTCConnectionMethod.LocalAP)

        logger.info("Connecting to Go2 at %s via go2-webrtc-connect (mode=%s, token=%s)...",
                     self.robot_ip, self.connection_method,
                     "set" if self.token else "none")

        self._conn = Go2WebRTCConnection(
            method,
            serialNumber=self.token or None,
            ip=self.robot_ip,
        )
        await self._conn.connect()

        self._connected_event.set()
        self.state = RobotState.STANDING

        # Register disconnect handler if the library supports it
        if hasattr(self._conn, 'on_disconnect'):
            self._conn.on_disconnect = self._handle_disconnect

        # Subscribe to robot state updates so battery/position/velocity are tracked.
        # The community lib handles data channel messages internally via pub_sub.
        pub_sub = getattr(self._conn, 'pub_sub', None)
        if pub_sub and hasattr(pub_sub, 'subscribe'):
            try:
                pub_sub.subscribe("rt/lf/state", self._handle_state_update)
                logger.info("Subscribed to robot state updates via pub_sub")
            except Exception as e:
                logger.warning("Failed to subscribe to state updates: %s", e)

        # Start video frame reader if camera is in webrtc mode
        if self._camera and self._conn.pc:
            self._video_task = asyncio.create_task(self._read_video_frames())

        logger.info("Connected to Go2 via go2-webrtc-connect")

    async def _connect_aiortc(self):
        """Fallback: connect using raw aiortc (our original implementation)."""
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
            import aiohttp
        except ImportError:
            raise ImportError(
                "No WebRTC library available. Install one:\n"
                "  pip install go2-webrtc-connect   (recommended)\n"
                "  pip install aiortc aiohttp        (fallback)"
            )

        logger.info("Connecting to Go2 at %s via aiortc...", self.robot_ip)

        self._pc = RTCPeerConnection()
        self._dc = self._pc.createDataChannel("data", ordered=True)

        # Request video track if camera needs WebRTC frames
        if self._camera:
            self._pc.addTransceiver("video", direction="recvonly")

        @self._dc.on("open")
        def on_open():
            logger.info("WebRTC data channel opened")
            self._connected_event.set()

        @self._dc.on("close")
        def on_close():
            self._handle_disconnect()

        @self._dc.on("message")
        def on_message(message):
            self._handle_message(message)

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        # WebRTC signaling endpoint. Port 8081 is the default for Go2 Pro
        # on stock firmware. Some firmware versions or models may use 9991.
        # The community library (go2-webrtc-connect) handles this internally.
        signal_url = f"http://{self.robot_ip}:8081/offer"
        async with aiohttp.ClientSession() as session:
            payload = {
                "sdp": self._pc.localDescription.sdp,
                "type": self._pc.localDescription.type,
                "token": self.token,
            }
            async with session.post(signal_url, json=payload) as resp:
                if resp.status != 200:
                    raise ConnectionError(
                        f"WebRTC signaling failed (HTTP {resp.status}). "
                        f"Is the robot on and connected to the network?"
                    )
                answer = await resp.json()

        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

        await asyncio.sleep(1.0)
        self._connected_event.set()
        self.state = RobotState.STANDING

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start video frame reader if camera is in webrtc mode
        if self._camera:
            self._video_task = asyncio.create_task(self._read_video_frames())

        logger.info("Connected to Go2 via aiortc")

    def _handle_disconnect(self):
        """Called when the WebRTC connection drops."""
        logger.warning("WebRTC data channel CLOSED — connection lost")
        self._connected_event.clear()
        if self._on_disconnect_callback:
            self._on_disconnect_callback()

    async def disconnect(self):
        """Gracefully disconnect from the robot."""
        if self._video_task:
            self._video_task.cancel()
            self._video_task = None

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        if self._use_community_lib and self._conn:
            try:
                await self._conn.disconnect()
            except Exception:
                pass
            self._conn = None
        elif self._pc:
            await self._pc.close()
            self._pc = None
            self._dc = None

        self._connected_event.clear()
        self.state = RobotState.DISCONNECTED
        logger.info("Disconnected from Go2")

    def _send_command(self, api_id: int, parameter: Optional[dict] = None) -> bool:
        """
        Send a sport mode command over the WebRTC data channel.

        Returns True if the command was sent, False if the connection is down.
        """
        if self._use_community_lib and self._conn:
            try:
                # The community lib's pub_sub.publish() expects a topic and a
                # dict message. It handles JSON serialization and data channel
                # framing internally.
                pub_sub = getattr(self._conn, 'pub_sub', None)
                if pub_sub:
                    msg = {
                        "header": {"identity": {"id": api_id, "api_id": api_id}},
                        "parameter": parameter or {},
                    }
                    pub_sub.publish("rt/api/sport/request", msg)
                    return True
                # Fallback: raw data channel access (older library versions)
                dc = getattr(self._conn, 'datachannel', None) or getattr(self._conn, 'dc', None)
                if dc:
                    msg = {
                        "header": {"identity": {"id": api_id, "api_id": api_id}},
                        "parameter": json.dumps(parameter or {}),
                    }
                    dc.send(json.dumps(msg))
                    return True
                logger.warning("Cannot find data channel on community lib connection")
                return False
            except Exception as e:
                logger.error("Failed to send command via community lib: %s", e)
                self._connected_event.clear()
                return False

        # Fallback: raw aiortc data channel
        msg = {
            "header": {"identity": {"id": api_id, "api_id": api_id}},
            "parameter": json.dumps(parameter or {}),
        }
        msg_str = json.dumps(msg)

        if not self._dc or self._dc.readyState != "open":
            logger.warning("Data channel not open, cannot send command (api_id=%d)", api_id)
            self._connected_event.clear()
            return False

        try:
            self._dc.send(msg_str)
            return True
        except Exception as e:
            logger.error("Failed to send command (api_id=%d): %s", api_id, e)
            self._connected_event.clear()
            return False

    def _handle_state_update(self, message):
        """Handle state updates from the community lib's pub_sub."""
        try:
            # The community lib may pass the message as a dict or JSON string
            if isinstance(message, str):
                data = json.loads(message)
            elif isinstance(message, dict):
                data = message
            else:
                return
            # State data may be nested under "data" key (same as aiortc path)
            if "data" in data and isinstance(data["data"], dict):
                data = data["data"]
            self._update_status(data)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.debug("Failed to parse state update: %s", e)

    def _handle_message(self, message: str):
        """Handle incoming state messages from the robot (aiortc fallback)."""
        try:
            data = json.loads(message)
            if "data" in data:
                self._update_status(data["data"])
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.debug("Failed to parse state message: %s", e)

    def _update_status(self, state_data: dict):
        """Update internal status from robot state data."""
        def _safe_list(val, idx, default=0):
            if isinstance(val, list) and len(val) > idx:
                return val[idx]
            return default

        position = state_data.get("position", [0, 0, 0])
        velocity = state_data.get("velocity", [0, 0, 0])
        imu_rpy = state_data.get("imu", {}).get("rpy", [0, 0, 0])

        self._status = Go2Status(
            state=self.state,
            battery_percent=state_data.get("bms", {}).get("soc", 0),
            position=(_safe_list(position, 0), _safe_list(position, 1), _safe_list(position, 2)),
            velocity=(_safe_list(velocity, 0), _safe_list(velocity, 1), _safe_list(velocity, 2)),
            imu_rpy=(_safe_list(imu_rpy, 0), _safe_list(imu_rpy, 1), _safe_list(imu_rpy, 2)),
            timestamp=time.time(),
        )

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep the connection alive."""
        try:
            while True:
                self._send_command(SportAPI.HEART)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def _read_video_frames(self):
        """
        Read video frames from the WebRTC video track and push to camera.

        This is how the camera works in STA-L mode: instead of UDP multicast
        (which doesn't cross WiFi routers), frames come through the same
        WebRTC connection used for control commands.
        """
        try:
            # Find the video track on the peer connection.
            # The track may not be available immediately after connect() returns
            # because ICE negotiation and track setup are asynchronous. Retry
            # with backoff to avoid a race condition.
            track = None
            pc = self._conn.pc if self._conn else self._pc
            if not pc:
                logger.warning("No peer connection for video frames")
                return

            for attempt in range(10):
                for transceiver in pc.getTransceivers():
                    if (transceiver.receiver and transceiver.receiver.track
                            and transceiver.receiver.track.kind == "video"):
                        track = transceiver.receiver.track
                        break
                if track:
                    break
                wait_time = min(0.5 * (attempt + 1), 3.0)
                logger.info("Waiting for WebRTC video track (attempt %d/10, %.1fs)...",
                            attempt + 1, wait_time)
                await asyncio.sleep(wait_time)

            if not track:
                logger.warning("No video track found on WebRTC connection after 10 attempts")
                return

            logger.info("WebRTC video track found — streaming frames to camera")
            frame_count = 0
            while self.connected:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                    # Convert av.VideoFrame to numpy BGR array
                    img = frame.to_ndarray(format="bgr24")
                    self._camera.push_frame(img)
                    frame_count += 1
                    if frame_count == 1:
                        logger.info("First WebRTC video frame received (%dx%d)",
                                     img.shape[1], img.shape[0])
                except asyncio.TimeoutError:
                    logger.warning("WebRTC video frame timeout — still waiting")
                    continue
                except Exception as e:
                    if "MediaStreamError" in type(e).__name__:
                        logger.warning("WebRTC video track ended")
                        break
                    logger.error("WebRTC video frame error: %s", e)
                    break

            logger.info("WebRTC video reader stopped (received %d frames)", frame_count)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("WebRTC video reader failed: %s", e)

    @property
    def status(self) -> Go2Status:
        """Get the latest robot status."""
        return self._status

    # ---- High-Level Movement Commands ----

    @property
    def battery_percent(self) -> float:
        """Get the latest battery level."""
        return self._status.battery_percent

    async def stand_up(self):
        """Command the robot to stand up."""
        logger.info("Standing up")
        if self._send_command(SportAPI.STAND_UP):
            self.state = RobotState.STANDING
        else:
            logger.error("stand_up command failed to send")
        await asyncio.sleep(1.0)

    async def sit_down(self):
        """Command the robot to sit/lie down."""
        logger.info("Sitting down")
        if self._send_command(SportAPI.STAND_DOWN):
            self.state = RobotState.SITTING
        else:
            logger.error("sit_down command failed to send")
        await asyncio.sleep(1.0)

    async def stop(self):
        """Stop all movement (robot stays standing)."""
        logger.info("Stopping")
        if self._send_command(SportAPI.STOP_MOVE):
            self.state = RobotState.STANDING
        else:
            logger.error("stop command failed to send")

    async def emergency_stop(self):
        """Emergency soft stop — robot goes limp (damped mode)."""
        logger.warning("EMERGENCY STOP — damping motors")
        self._send_command(SportAPI.DAMP)
        self.state = RobotState.DAMPED

    async def move(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0):
        """
        Send velocity command to the robot.

        Args:
            vx: Forward velocity in m/s (positive = forward).
            vy: Lateral velocity in m/s (positive = left).
            vyaw: Rotational velocity in rad/s (positive = counter-clockwise).
        """
        vx = max(-self.max_linear_vel, min(self.max_linear_vel, vx))
        vy = max(-self.max_linear_vel, min(self.max_linear_vel, vy))
        vyaw = max(-self.max_angular_vel, min(self.max_angular_vel, vyaw))

        if not self._send_command(SportAPI.MOVE, {"x": vx, "y": vy, "z": vyaw}):
            logger.error("move command failed to send")
            return
        self.state = RobotState.MOVING

    async def _interruptible_sleep(self, duration: float, step: float = 0.1):
        """
        Sleep in small steps so movement can be interrupted by Ctrl+C or disconnect.

        Checks connection status every `step` seconds. If connection is lost
        or the current asyncio task is cancelled, returns early so the caller
        can stop gracefully.
        """
        deadline = time.monotonic() + duration
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            chunk = min(step, remaining)
            try:
                await asyncio.sleep(chunk)
            except asyncio.CancelledError:
                logger.info("Movement interrupted by task cancellation")
                return
            if not self.connected:
                logger.warning("Connection lost during movement — aborting")
                return

    async def move_forward(self, distance: float, speed: float = 0.3):
        """
        Move forward a specified distance.

        Movement is interruptible — checked every 100ms so Ctrl+C or
        connection loss will stop the robot promptly.
        """
        speed = min(speed, self.max_linear_vel)
        duration = abs(distance) / speed if speed > 0 else 0
        duration = min(duration, 10.0)  # Safety cap

        logger.info("Moving forward %.2fm at %.2fm/s (%.1fs)", distance, speed, duration)
        await self.move(vx=speed)
        await self._interruptible_sleep(duration)
        await self.stop()

    async def move_backward(self, distance: float, speed: float = 0.3):
        """
        Move backward a specified distance.

        Movement is interruptible — checked every 100ms so Ctrl+C or
        connection loss will stop the robot promptly.
        """
        speed = min(speed, self.max_linear_vel)
        duration = abs(distance) / speed if speed > 0 else 0
        duration = min(duration, 10.0)  # Safety cap

        logger.info("Moving backward %.2fm at %.2fm/s (%.1fs)", distance, speed, duration)
        await self.move(vx=-speed)
        await self._interruptible_sleep(duration)
        await self.stop()

    async def turn(self, angle_degrees: float, speed: float = 0.5):
        """
        Turn by a specified angle.

        Args:
            angle_degrees: Positive = left (CCW), negative = right (CW).
            speed: Rotational speed in rad/s.

        Movement is interruptible — checked every 100ms.
        """
        angle_rad = math.radians(angle_degrees)
        speed = min(speed, self.max_angular_vel)
        duration = abs(angle_rad) / speed if speed > 0 else 0
        duration = min(duration, 10.0)  # Safety cap
        direction = 1.0 if angle_degrees > 0 else -1.0

        logger.info("Turning %.1f degrees at %.2f rad/s", angle_degrees, speed)
        await self.move(vyaw=direction * speed)
        await self._interruptible_sleep(duration)
        await self.stop()

    async def recovery_stand(self):
        """Attempt to recover to standing position (e.g., after a fall)."""
        logger.info("Recovery stand")
        if self._send_command(SportAPI.RECOVERY_STAND):
            self.state = RobotState.STANDING
        else:
            logger.error("recovery_stand command failed to send")
        await asyncio.sleep(2.0)
