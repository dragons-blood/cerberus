#!/usr/bin/env python3
"""
Preflight check — verifies your setup before running the robot.

Run this after setup_jetson.sh to make sure everything is ready:
    source venv/bin/activate
    python scripts/preflight_check.py

It checks:
  1. Python version
  2. Required Python packages
  3. Gemini API key
  4. OpenCV GStreamer support
  5. Network connectivity to the Go2
  6. Internet access (for Gemini API)
  7. Config file validity
"""

import os
import sys
import subprocess
from pathlib import Path

# Make sure we can import from the repo
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

# Load .env so we can check the API key
from dotenv import load_dotenv
load_dotenv(REPO_DIR / "config" / ".env")

PASS = "  PASS"
FAIL = "  FAIL"
WARN = "  WARN"

passed = 0
failed = 0
warned = 0


def check(name, ok, fail_msg="", warn_only=False):
    global passed, failed, warned
    if ok:
        print(f"{PASS}  {name}")
        passed += 1
    elif warn_only:
        print(f"{WARN}  {name}")
        if fail_msg:
            print(f"        {fail_msg}")
        warned += 1
    else:
        print(f"{FAIL}  {name}")
        if fail_msg:
            print(f"        {fail_msg}")
        failed += 1


def main():
    global passed, failed, warned

    print()
    print("=" * 55)
    print("  Gemini Robot Dog — Preflight Check")
    print("=" * 55)
    print()

    # --- 1. Python version ---
    print("[1/7] Python version")
    v = sys.version_info
    check(
        f"Python {v.major}.{v.minor}.{v.micro}",
        v >= (3, 10),
        "Python 3.10+ required. JetPack 6.x includes it.",
    )
    print()

    # --- 2. Required packages ---
    print("[2/7] Python packages")
    packages = {
        "google.genai": "google-genai (Gemini SDK)",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "numpy": "numpy",
        "aiohttp": "aiohttp",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
    }
    for module, name in packages.items():
        try:
            __import__(module)
            check(name, True)
        except ImportError:
            check(name, False, f"pip install {name.split('(')[0].strip()}")

    # WebRTC: prefer go2-webrtc-connect, fall back to aiortc
    has_go2_lib = False
    try:
        __import__("go2_webrtc_driver")
        has_go2_lib = True
        check("go2-webrtc-connect (WebRTC)", True)
    except ImportError:
        check("go2-webrtc-connect (WebRTC)", False,
              "pip install go2-webrtc-connect  (recommended)", warn_only=True)
    try:
        __import__("aiortc")
        check("aiortc (WebRTC fallback)", True)
    except ImportError:
        if has_go2_lib:
            check("aiortc (WebRTC fallback)", False,
                  "pip install aiortc  (optional fallback)", warn_only=True)
        else:
            check("aiortc (WebRTC fallback)", False,
                  "No WebRTC library! pip install go2-webrtc-connect")
    print()

    # --- 3. Gemini API key ---
    print("[3/7] Gemini API key")
    env_file = REPO_DIR / "config" / ".env"
    check("config/.env exists", env_file.exists(), "Run: cp config/.env.example config/.env")

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    has_key = bool(api_key) and api_key != "your_gemini_api_key_here"
    check(
        "GOOGLE_API_KEY is set",
        has_key,
        "Edit config/.env and add your key from https://aistudio.google.com/apikey",
    )

    if has_key:
        # Try a quick API call to verify the key works
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            # Just list models — lightweight check
            models = client.models.list()
            check("API key is valid (connected to Gemini)", True)
        except Exception as e:
            check("API key is valid", False, f"API error: {e}")
    print()

    # --- 4. OpenCV + GStreamer ---
    print("[4/7] OpenCV + GStreamer")
    try:
        import cv2
        check(f"OpenCV {cv2.__version__}", True)
        build_info = cv2.getBuildInformation()
        has_gst = "GStreamer" in build_info and "YES" in build_info.split("GStreamer")[1][:50]
        check(
            "OpenCV has GStreamer support",
            has_gst,
            "Camera may not work. Rebuild OpenCV with -DWITH_GSTREAMER=ON\n"
            "        or set camera.source to 'usb' in config/robot_config.yaml",
            warn_only=True,
        )
    except ImportError:
        check("OpenCV", False, "pip install opencv-python")

    # Check gst-launch-1.0
    gst_ok = subprocess.run(
        ["gst-launch-1.0", "--version"],
        capture_output=True,
    ).returncode == 0 if subprocess.run(["which", "gst-launch-1.0"], capture_output=True).returncode == 0 else False
    check("gst-launch-1.0 available", gst_ok, "sudo apt install gstreamer1.0-plugins-base", warn_only=True)
    print()

    # --- 5. Network: Go2 reachable ---
    print("[5/7] Go2 network connectivity")
    import yaml as _yaml
    config_path = REPO_DIR / "config" / "robot_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = _yaml.safe_load(f)
        robot_ip = config.get("unitree", {}).get("robot_ip", "192.168.12.1")
        conn_method = config.get("unitree", {}).get("connection_method", "ap")
        robot_iface = config.get("jetson", {}).get("robot_interface", "")
    else:
        robot_ip = "192.168.12.1"
        conn_method = "ap"
        robot_iface = ""

    mode_label = "STA-L (shared WiFi)" if conn_method == "sta" else "AP (robot hotspot)"
    print(f"      Connection mode: {mode_label}")
    if robot_iface:
        print(f"      Robot interface: {robot_iface}")

    # Ping the robot (1 packet, 2 second timeout)
    # Use -I <interface> to force traffic through the correct adapter
    # (critical in AP mode where Ethernet goes to internet, WiFi goes to Go2)
    ping_cmd = ["ping", "-c", "1", "-W", "2"]
    if robot_iface:
        ping_cmd += ["-I", robot_iface]
    ping_cmd.append(robot_ip)
    ping_result = subprocess.run(ping_cmd, capture_output=True)
    check(
        f"Go2 reachable at {robot_ip}" + (f" (via {robot_iface})" if robot_iface else ""),
        ping_result.returncode == 0,
        f"Cannot ping {robot_ip}. Is the Go2 on? Is the Jetson on its WiFi?\n"
        f"        Connect: nmcli device wifi connect GO2_XXXX password the-password\n"
        f"        Check interface: ip link show {robot_iface}" if robot_iface else
        f"Cannot ping {robot_ip}. Is the Go2 on? Is the Jetson on its WiFi?\n"
        f"        Connect: nmcli device wifi connect GO2_XXXX password the-password",
        warn_only=True,
    )
    print()

    # --- 6. Internet access ---
    print("[6/7] Internet access (for Gemini API)")
    try:
        import urllib.request
        import urllib.error
        try:
            urllib.request.urlopen("https://generativelanguage.googleapis.com/", timeout=5)
        except urllib.error.HTTPError:
            # HTTP error (404, 403, etc.) still means internet is working —
            # we reached the server, it just didn't like the bare URL.
            pass
        check("Internet access to Gemini API", True)
    except Exception:
        if conn_method == "sta":
            internet_hint = (
                "No internet. In STA-L mode your WiFi should provide internet.\n"
                "        Check that the Jetson is connected to your WiFi:\n"
                "          nmcli device wifi list\n"
                "        Check DNS is working:\n"
                "          nslookup generativelanguage.googleapis.com"
            )
        else:
            internet_hint = (
                "No internet. In AP mode the Jetson's WiFi is connected to the Go2's\n"
                "        hotspot, so you need a second connection for internet:\n"
                "        - Plug in an Ethernet cable to your router, OR\n"
                "        - Use a USB WiFi adapter for internet\n"
                "        Or switch to STA-L mode (see README) to use one network for everything."
            )
        check(
            "Internet access to Gemini API",
            False,
            internet_hint,
        )
    print()

    # --- 7. Config file ---
    print("[7/7] Configuration")
    config_ok = config_path.exists()
    check("config/robot_config.yaml exists", config_ok, "File is missing from repo")
    if config_ok:
        try:
            with open(config_path) as f:
                config = _yaml.safe_load(f)
            for section in ["gemini", "unitree", "camera"]:
                check(f"Config section '{section}'", section in config, f"Missing '{section}' in config")
        except Exception as e:
            check("Config file valid YAML", False, str(e))
    print()

    # --- Summary ---
    total = passed + failed + warned
    print("=" * 55)
    if failed == 0:
        if warned == 0:
            print(f"  ALL {total} CHECKS PASSED!")
            print()
            print("  You're ready to run:")
            print("    python -m src.robot --interactive")
        else:
            print(f"  {passed}/{total} passed, {warned} warnings")
            print()
            print("  Warnings are non-fatal but may cause issues.")
            print("  You can try running: python -m src.robot --interactive")
    else:
        print(f"  {failed} CHECK(S) FAILED ({passed} passed, {warned} warnings)")
        print()
        print("  Fix the FAIL items above before running the robot.")
    print("=" * 55)
    print()

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
