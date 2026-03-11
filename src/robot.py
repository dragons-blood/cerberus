"""
Main robot controller — Gemini Robotics + Jetson Orin + Unitree Go2 Pro.

This is the central orchestrator that ties together:
- Camera capture (Jetson Orin / Go2 front camera)
- Gemini Robotics-ER for embodied reasoning (perception + planning)
- Unitree Go2 sport mode for locomotion

The main control loop:
  1. Capture a camera frame
  2. Send it to Gemini with the current task instruction
  3. Receive a plan (list of actions)
  4. Execute each action on the Go2
  5. Repeat

Usage:
    python -m src.robot --instruction "follow the person"
    python -m src.robot --instruction "go to the red cone"
    python -m src.robot --interactive  # Interactive mode (type commands)
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.gemini.robotics_client import GeminiRoboticsClient, RobotAction
from src.guardrails import GuardrailConfig, validate_plan
from src.unitree.go2_controller import Go2Controller
from src.vision.camera import Camera

logger = logging.getLogger(__name__)

# Battery thresholds
BATTERY_WARN_PERCENT = 20.0
BATTERY_CRITICAL_PERCENT = 10.0

REQUIRED_CONFIG_KEYS = {
    "gemini": ["model_id", "thinking_budget", "max_output_tokens"],
    "unitree": ["robot_ip", "max_linear_velocity", "max_angular_velocity"],
    "camera": ["source", "width", "height", "fps"],
}


def load_config(config_path: str = "config/robot_config.yaml") -> dict:
    """Load robot configuration from YAML."""
    path = Path(__file__).parent.parent / config_path
    if not path.exists():
        print(f"\nERROR: Config file not found: {path}")
        print("Make sure you're running from the ROBOT/ directory.")
        sys.exit(1)
    with open(path) as f:
        config = yaml.safe_load(f)
    # Validate required keys
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in config:
            print(f"\nERROR: Missing '{section}' section in {config_path}")
            sys.exit(1)
        for key in keys:
            if key not in config[section]:
                print(f"\nERROR: Missing '{section}.{key}' in {config_path}")
                sys.exit(1)
    return config


def preflight_checks():
    """
    Run basic checks before starting the robot.
    Catches common mistakes early with clear error messages.
    """
    errors = []

    # 1. Check API key
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        env_file = Path(__file__).parent.parent / "config" / ".env"
        if not env_file.exists():
            errors.append(
                "No config/.env file found.\n"
                "  Run: cp config/.env.example config/.env\n"
                "  Then edit config/.env and add your GOOGLE_API_KEY\n"
                "  Get a free key at: https://aistudio.google.com/apikey"
            )
        else:
            errors.append(
                "GOOGLE_API_KEY is not set in config/.env\n"
                "  Edit config/.env and add your Gemini API key.\n"
                "  Get a free key at: https://aistudio.google.com/apikey"
            )

    # 2. Check OpenCV
    try:
        import cv2
    except ImportError:
        errors.append(
            "OpenCV is not installed.\n"
            "  Run: source venv/bin/activate && pip install opencv-python"
        )

    # 3. Check WebRTC library (go2-webrtc-connect preferred, aiortc as fallback)
    has_webrtc = False
    try:
        import go2_webrtc_driver  # noqa: F401
        has_webrtc = True
    except ImportError:
        pass
    if not has_webrtc:
        try:
            import aiortc  # noqa: F401
            has_webrtc = True
        except ImportError:
            pass
    if not has_webrtc:
        errors.append(
            "No WebRTC library installed. Install one:\n"
            "  pip install go2-webrtc-connect   (recommended)\n"
            "  pip install aiortc               (fallback)"
        )

    if errors:
        print("\n" + "=" * 50)
        print("  PREFLIGHT CHECK FAILED")
        print("=" * 50)
        for i, err in enumerate(errors, 1):
            print(f"\n  {i}. {err}")
        print("\n" + "=" * 50)
        print("  Fix the above and try again.")
        print("  For full diagnostics: python scripts/preflight_check.py")
        print("=" * 50 + "\n")
        sys.exit(1)


class Robot:
    """
    Main robot orchestrator.

    Connects Gemini's embodied reasoning to the Go2's locomotion,
    with the camera as the perception bridge.
    """

    def __init__(self, config: dict):
        self.config = config
        self.guardrails = GuardrailConfig()

        # Initialize subsystems
        self.camera = Camera(
            source=config["camera"]["source"],
            width=config["camera"]["width"],
            height=config["camera"]["height"],
            fps=config["camera"]["fps"],
            gstreamer_address=config["camera"]["gstreamer_address"],
            gstreamer_port=config["camera"]["gstreamer_port"],
        )

        self.gemini = GeminiRoboticsClient(
            model_id=config["gemini"]["model_id"],
            thinking_budget=config["gemini"]["thinking_budget"],
            max_output_tokens=config["gemini"]["max_output_tokens"],
        )

        unitree_cfg = config["unitree"]
        # Token: config file, then .env, then empty string
        token = unitree_cfg.get("token", "") or os.environ.get("UNITREE_TOKEN", "")
        self.go2 = Go2Controller(
            robot_ip=unitree_cfg["robot_ip"],
            max_linear_vel=unitree_cfg["max_linear_velocity"],
            max_angular_vel=unitree_cfg["max_angular_velocity"],
            token=token,
        )

        self._running = False

    async def start(self):
        """Initialize all subsystems."""
        logger.info("Starting robot systems...")

        # Start camera (with timeout so we don't hang if Go2 isn't streaming)
        try:
            self.camera.start()
        except RuntimeError as e:
            logger.error("Camera failed to start: %s", e)
            print(f"\nERROR: {e}")
            print("The robot cannot operate without a camera. Exiting.\n")
            sys.exit(1)
        logger.info("Camera ready (source=%s)", self.config["camera"]["source"])

        # Connect to Go2 (with timeout)
        try:
            await asyncio.wait_for(self.go2.connect(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("Go2 connection timed out after 15s")
            print("\nERROR: Could not connect to Go2 at %s (timed out)" % self.config["unitree"]["robot_ip"])
            print("Check that the Go2 is on and Jetson is on its WiFi.\n")
            self.camera.stop()
            sys.exit(1)
        except ConnectionError as e:
            logger.error("Go2 connection failed: %s", e)
            print(f"\nERROR: {e}\n")
            self.camera.stop()
            sys.exit(1)
        logger.info("Go2 connected (ip=%s)", self.config["unitree"]["robot_ip"])

        # Register disconnect callback — auto-stop robot if WebRTC drops
        def _on_disconnect():
            logger.critical("Go2 WebRTC connection lost — stopping robot")
            self._running = False

        self.go2._on_disconnect_callback = _on_disconnect

        # Stand up
        await self.go2.stand_up()

        self._running = True
        logger.info("Robot is ready!")

    async def stop(self):
        """Gracefully shut down all subsystems."""
        logger.info("Shutting down robot...")
        self._running = False

        await self.go2.stop()
        await self.go2.sit_down()
        await self.go2.disconnect()
        self.camera.stop()

        logger.info("Robot shut down complete")

    async def emergency_stop(self):
        """
        Immediate emergency stop — motors go limp.

        This is the nuclear option: bypasses graceful shutdown and sends DAMP
        directly. The robot will collapse in place. Use when the robot is about
        to hit something or someone, or when the software has lost control.
        """
        logger.critical("!!! EMERGENCY STOP !!!")
        self._running = False
        await self.go2.emergency_stop()
        self.camera.stop()
        logger.critical("Robot is in damped mode — motors are limp")

    async def execute_action(self, action: RobotAction):
        """
        Execute a single robot action on the Go2.

        Translates Gemini's planned actions into Go2 sport mode commands.
        """
        name = action.action
        params = action.parameters

        logger.info("Executing: %s %s", name, params)

        if name == "move_forward":
            await self.go2.move_forward(
                distance=params.get("distance", 0.5),
                speed=params.get("speed", 0.3),
            )
        elif name == "move_backward":
            await self.go2.move_backward(
                distance=params.get("distance", 0.5),
                speed=params.get("speed", 0.3),
            )
        elif name == "turn_left":
            await self.go2.turn(angle_degrees=abs(params.get("angle", 45)))
        elif name == "turn_right":
            await self.go2.turn(angle_degrees=-abs(params.get("angle", 45)))
        elif name == "stop":
            await self.go2.stop()
        elif name == "stand_up":
            await self.go2.stand_up()
        elif name == "sit_down":
            await self.go2.sit_down()
        elif name == "look_at":
            # Approximate "look at" by turning toward the target's x-coordinate
            target_x = params.get("x", 0.5)
            # If target is left of center, turn left; right of center, turn right
            offset = target_x - 0.5
            angle = -offset * 60  # Scale: 0.5 offset ≈ 30 degrees
            if abs(angle) > 5:
                await self.go2.turn(angle_degrees=angle)
        elif name == "wait":
            seconds = min(params.get("seconds", 1.0), 10.0)
            await self.go2._interruptible_sleep(seconds)
        elif name == "speak":
            # TTS would be handled by a separate module
            logger.info("ROBOT SAYS: %s", params.get("message", ""))
        else:
            logger.warning("Unknown action: %s", name)

    async def run_instruction(self, instruction: str):
        """
        Execute a single natural language instruction.

        Captures a frame, sends it to Gemini for planning, then executes
        the resulting action sequence on the Go2.
        """
        logger.info("Instruction: %s", instruction)

        # Capture current scene
        frame = self.camera.get_frame()
        if frame is None:
            logger.error("No camera frame available")
            return

        # Plan actions
        actions = self.gemini.plan_actions(frame, instruction)
        if not actions:
            logger.warning("Gemini returned no actions for: %s", instruction)
            return

        # Validate through guardrails before executing
        actions, warnings = validate_plan(actions, self.guardrails)
        for w in warnings:
            logger.warning("Guardrail: %s", w)

        logger.info("Planned %d actions (after guardrails):", len(actions))
        for i, action in enumerate(actions):
            logger.info("  [%d] %s %s", i + 1, action.action, action.parameters)

        # Execute action sequence
        for action in actions:
            if not self._running:
                break
            await self.execute_action(action)

        logger.info("Instruction complete: %s", instruction)

    async def run_continuous(self, instruction: str, interval: float = 3.0):
        """
        Continuously execute an instruction in a perception-action loop.

        The robot will repeatedly capture frames, re-plan, and act.
        This is useful for ongoing tasks like "follow the person" or
        "patrol the area".

        Safety features:
        - Watchdog: if no successful Gemini plan in watchdog_timeout seconds,
          the robot stops and waits (prevents runaway if internet drops).
        - Battery: warns at 20%, auto-stops at 10%.
        """
        watchdog_timeout = self.config.get("safety", {}).get("watchdog_timeout", 30.0)
        logger.info("Starting continuous mode: %s (interval=%.1fs, watchdog=%.0fs)",
                     instruction, interval, watchdog_timeout)

        last_successful_plan = time.time()
        battery_warned = False

        while self._running:
            # --- Battery check ---
            battery = self.go2.battery_percent
            if battery > 0:  # 0 means no data yet
                if battery <= BATTERY_CRITICAL_PERCENT:
                    logger.critical("Battery critically low (%.0f%%) — stopping robot", battery)
                    print(f"\nBATTERY CRITICAL: {battery:.0f}%% — robot is stopping for safety.")
                    await self.go2.stop()
                    await self.go2.sit_down()
                    self._running = False
                    break
                elif battery <= BATTERY_WARN_PERCENT and not battery_warned:
                    logger.warning("Battery low (%.0f%%) — consider stopping soon", battery)
                    print(f"\nWARNING: Battery at {battery:.0f}%%")
                    battery_warned = True

            # --- Capture and plan ---
            frame = self.camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.5)
                continue

            actions = self.gemini.plan_actions(frame, instruction)
            actions, _ = validate_plan(actions, self.guardrails)

            if actions:
                last_successful_plan = time.time()
                for action in actions:
                    if not self._running:
                        break
                    await self.execute_action(action)
            else:
                # --- Watchdog: no successful plan for too long ---
                since_last = time.time() - last_successful_plan
                if since_last > watchdog_timeout:
                    logger.warning("Watchdog: no successful plan for %.0fs — stopping robot",
                                   since_last)
                    await self.go2.stop()
                    # Reset watchdog so we don't spam stops
                    last_successful_plan = time.time()

            await asyncio.sleep(interval)

    async def run_interactive(self):
        """
        Interactive mode — type instructions in the terminal.

        Special commands:
            !scene     — Describe the current scene
            !detect    — Detect objects in view
            !status    — Show robot status
            !quit      — Shut down
        """
        logger.info("Interactive mode. Type instructions or !help for commands.")
        print("\n--- Interactive Robot Control ---")
        print("Type a natural language instruction, or:")
        print("  !scene   - Describe what the robot sees")
        print("  !detect  - Detect objects")
        print("  !status  - Robot status")
        print("  !quit    - Shut down\n")

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                user_input = await loop.run_in_executor(None, lambda: input(">> "))
            except EOFError:
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input == "!quit":
                break
            elif user_input == "!scene":
                frame = self.camera.get_frame()
                if frame is not None:
                    desc = self.gemini.describe_scene(frame)
                    print(f"Scene: {desc}")
                else:
                    print("No camera frame available")
            elif user_input == "!detect":
                frame = self.camera.get_frame()
                if frame is not None:
                    objects = self.gemini.detect_objects(frame)
                    for obj in objects:
                        print(f"  {obj.label} at ({obj.x:.2f}, {obj.y:.2f})")
                    if not objects:
                        print("  No objects detected")
                else:
                    print("No camera frame available")
            elif user_input == "!status":
                s = self.go2.status
                print(f"  State: {s.state.value}")
                print(f"  Battery: {s.battery_percent:.0f}%")
                print(f"  Position: {s.position}")
                print(f"  Velocity: {s.velocity}")
            else:
                await self.run_instruction(user_input)


async def main():
    parser = argparse.ArgumentParser(description="Gemini Robotics + Go2 Pro Robot Controller")
    parser.add_argument("--instruction", "-i", type=str, help="Single instruction to execute")
    parser.add_argument("--continuous", "-c", action="store_true",
                        help="Run instruction continuously in a perception-action loop")
    parser.add_argument("--interactive", action="store_true", help="Interactive command mode")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Interval between perception-action cycles in continuous mode (seconds)")
    parser.add_argument("--config", type=str, default="config/robot_config.yaml",
                        help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Setup logging — console + persistent log file for post-incident review
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_format,
        datefmt=log_datefmt,
    )
    # Add file handler so robot behavior is recorded to disk
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "robot.log")
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=log_datefmt))
    logging.getLogger().addHandler(file_handler)

    # Load environment variables (inside main, not at import time)
    _env_path = Path(__file__).parent.parent / "config" / ".env"
    load_dotenv(_env_path)

    # --- Preflight: catch common mistakes before touching hardware ---
    preflight_checks()

    config = load_config(args.config)
    robot = Robot(config)

    # --- Two-stage kill switch ---
    # 1st Ctrl+C → graceful shutdown (stop, sit down, disconnect)
    # 2nd Ctrl+C → EMERGENCY STOP (motors go limp immediately)
    sigint_count = 0

    def signal_handler(sig, frame):
        nonlocal sigint_count
        sigint_count += 1

        if sigint_count == 1:
            logger.warning("Ctrl+C — graceful shutdown (press again for EMERGENCY STOP)")
            robot._running = False
        else:
            # Second Ctrl+C — emergency stop, no waiting
            logger.critical("Ctrl+C x2 — EMERGENCY STOP")
            # Schedule emergency stop via the event loop (thread-safe)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda: loop.create_task(robot.emergency_stop())
                    )
            except Exception:
                pass
            # Force exit after a brief delay if the loop is stuck
            def force_exit():
                time.sleep(2.0)
                logger.critical("Forced exit")
                os._exit(1)
            threading.Thread(target=force_exit, daemon=True).start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Optional: spacebar kill switch (background keyboard listener) ---
    # Capture the event loop reference for thread-safe scheduling
    main_loop = asyncio.get_event_loop()

    def _keyboard_kill_listener():
        """
        Listens for the spacebar as a hardware-style kill switch.
        Works when running on the Orin's desktop with a keyboard attached.
        Falls back gracefully if no terminal / no keyboard.
        """
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while robot._running:
                    ch = sys.stdin.read(1)
                    if ch == ' ':
                        logger.critical("SPACEBAR KILL SWITCH — EMERGENCY STOP")
                        robot._running = False
                        try:
                            main_loop.call_soon_threadsafe(
                                lambda: main_loop.create_task(robot.emergency_stop())
                            )
                        except Exception:
                            pass
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            # No terminal available (running headless, piped, etc.) — skip
            pass

    # Don't start keyboard listener in interactive mode (conflicts with input())
    if not args.interactive:
        kill_thread = threading.Thread(target=_keyboard_kill_listener, daemon=True)
        kill_thread.start()
        logger.info("Kill switch active: Ctrl+C (graceful) | Ctrl+C x2 (emergency) | Spacebar (emergency)")
    else:
        logger.info("Kill switch active: Ctrl+C (graceful) | Ctrl+C x2 (emergency)")

    # Gemini API timeout is enforced inside GeminiRoboticsClient._generate()
    # (default 30s). If Gemini doesn't respond, the robot stops that iteration.

    try:
        await robot.start()

        if args.interactive:
            await robot.run_interactive()
        elif args.instruction:
            if args.continuous:
                await robot.run_continuous(args.instruction, interval=args.interval)
            else:
                await robot.run_instruction(args.instruction)
        else:
            # Default to interactive mode
            await robot.run_interactive()
    finally:
        await robot.stop()


if __name__ == "__main__":
    asyncio.run(main())
