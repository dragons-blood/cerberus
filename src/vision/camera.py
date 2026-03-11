"""
Camera capture pipeline for Jetson Orin + Unitree Go2.

Supports multiple camera sources:
- GStreamer UDP multicast (Go2's front camera at 230.1.1.1:1720)
- USB cameras (via V4L2)
- CSI cameras (Jetson's MIPI CSI connector)

The camera runs in a background thread and provides the latest frame
on demand, so the main control loop never blocks on frame capture.
"""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """
    Threaded camera capture that always provides the latest frame.

    Usage:
        cam = Camera(source="gstreamer")
        cam.start()
        frame = cam.get_frame()  # numpy array (H, W, 3) BGR
        cam.stop()
    """

    # GStreamer pipeline for Go2 front camera (H264 over UDP multicast)
    GSTREAMER_PIPELINE = (
        "udpsrc address={address} port={port} "
        "! application/x-rtp,media=video,encoding-name=H264 "
        "! rtph264depay ! h264parse ! avdec_h264 "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink drop=1 sync=0"
    )

    # GStreamer pipeline for Jetson CSI camera (e.g., Raspberry Pi Camera v2)
    CSI_PIPELINE = (
        "nvarguscamerasrc sensor-id={sensor_id} "
        "! video/x-raw(memory:NVMM),width={width},height={height},"
        "framerate={fps}/1,format=NV12 "
        "! nvvidconv ! video/x-raw,format=BGRx "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink drop=1 sync=0"
    )

    def __init__(self, source: str = "gstreamer",
                 width: int = 1280, height: int = 720, fps: int = 30,
                 gstreamer_address: str = "230.1.1.1",
                 gstreamer_port: int = 1720,
                 usb_device: int = 0,
                 csi_sensor_id: int = 0):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.gstreamer_address = gstreamer_address
        self.gstreamer_port = gstreamer_port
        self.usb_device = usb_device
        self.csi_sensor_id = csi_sensor_id

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._start_time = 0.0

    def _build_pipeline(self) -> str | int:
        """Build the appropriate capture pipeline or device index."""
        if self.source == "gstreamer":
            return self.GSTREAMER_PIPELINE.format(
                address=self.gstreamer_address,
                port=self.gstreamer_port,
            )
        elif self.source == "csi":
            return self.CSI_PIPELINE.format(
                sensor_id=self.csi_sensor_id,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )
        elif self.source == "usb":
            return self.usb_device
        else:
            raise ValueError(f"Unknown camera source: {self.source}")

    def start(self, timeout: float = 10.0):
        """
        Start the camera capture thread.

        Args:
            timeout: Max seconds to wait for the camera to open. If the camera
                     doesn't respond in time (e.g., Go2 not streaming), raises
                     RuntimeError instead of hanging forever.
        """
        if self._running:
            logger.warning("Camera is already running")
            return

        pipeline = self._build_pipeline()
        logger.info("Opening camera: source=%s (timeout=%.0fs)", self.source, timeout)

        # Open camera in a background thread so we can enforce a timeout.
        # cv2.VideoCapture can block forever if the GStreamer multicast source
        # is unreachable (e.g., Go2 not on, WiFi not connected).
        cap_result = [None]
        cap_error = [None]

        def _open_camera():
            try:
                if isinstance(pipeline, str):
                    cap_result[0] = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                else:
                    cap = cv2.VideoCapture(pipeline)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    cap_result[0] = cap
            except Exception as e:
                cap_error[0] = e

        opener = threading.Thread(target=_open_camera, daemon=True)
        opener.start()
        opener.join(timeout=timeout)

        if opener.is_alive():
            # Camera open is still blocking — timed out
            raise RuntimeError(
                f"Camera timed out after {timeout:.0f}s (source={self.source}). "
                "The Go2 may not be streaming. Check that:\n"
                "  1. The Go2 is powered on\n"
                "  2. The Jetson is connected to the Go2's WiFi\n"
                "  3. You can ping the Go2: ping 192.168.12.1"
            )

        if cap_error[0]:
            raise RuntimeError(f"Camera error: {cap_error[0]}")

        self._cap = cap_result[0]

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera (source={self.source}). "
                "Check that the camera is connected and the pipeline is correct."
            )

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera started (source=%s, %dx%d @ %dfps)",
                     self.source, self.width, self.height, self.fps)

    def stop(self):
        """Stop the camera capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped (captured %d frames)", self._frame_count)

    def _capture_loop(self):
        """Background thread that continuously grabs frames."""
        consecutive_failures = 0
        max_failures = 50  # ~5 seconds of failures at 100ms backoff
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Camera: %d consecutive read failures — giving up", max_failures)
                    self._running = False
                    break
                # Backoff: 0.1s → 0.2s → 0.5s → 1.0s max
                backoff = min(0.1 * (2 ** (consecutive_failures // 10)), 1.0)
                logger.warning("Camera read failed (%d), retrying in %.1fs...",
                               consecutive_failures, backoff)
                time.sleep(backoff)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest camera frame.

        Returns:
            numpy array (H, W, 3) in BGR format, or None if no frame available.
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def get_frame_jpeg(self, quality: int = 85) -> Optional[bytes]:
        """
        Get the latest frame as JPEG bytes (for sending to Gemini API).

        Args:
            quality: JPEG quality (0-100).

        Returns:
            JPEG bytes or None.
        """
        frame = self.get_frame()
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    @property
    def actual_fps(self) -> float:
        """Calculate the actual capture FPS."""
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            return self._frame_count / elapsed
        return 0.0

    @property
    def is_running(self) -> bool:
        return self._running
