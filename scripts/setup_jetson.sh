#!/usr/bin/env bash
# =============================================================================
# Jetson Orin Setup Script for Gemini Robot Dog
#
# Run this on your Jetson Orin (Nano, NX, or AGX) to install everything.
# Assumes JetPack 6.x is already flashed (Ubuntu 22.04 + CUDA).
#
# Usage:
#   chmod +x scripts/setup_jetson.sh
#   ./scripts/setup_jetson.sh
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_DIR/venv"

echo "============================================"
echo "  Jetson Orin Setup for Gemini Robot Dog"
echo "============================================"
echo ""

# --- Check Python version ---
echo "[0/6] Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "  ERROR: Python 3.10 or newer is required (found $PYTHON_VERSION)"
    echo "  JetPack 6.x should include Python 3.10+. If you're on JetPack 5.x,"
    echo "  flash JetPack 6.x first: https://developer.nvidia.com/jetpack-sdk-62"
    exit 1
fi
echo "  Python $PYTHON_VERSION -- OK"
echo ""

# --- System packages ---
echo "[1/6] Installing system packages (this may take a few minutes)..."
sudo apt-get update -qq
if ! sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-dev \
    libgstreamer1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstrtspserver-1.0-dev \
    libopencv-dev \
    python3-opencv \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libsrtp2-dev \
    libvpx-dev \
    curl \
    git; then
    echo "  ERROR: Some system packages failed to install. Check output above."
    exit 1
fi
echo "  System packages installed"
echo ""

# --- Python virtual environment ---
echo "[2/6] Setting up Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR" --system-site-packages || { echo "  ERROR: Failed to create virtual environment"; exit 1; }
    echo "  Created virtual environment at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q || { echo "  ERROR: Failed to upgrade pip"; exit 1; }
echo ""

# --- Python dependencies ---
echo "[3/6] Installing Python dependencies..."
pip install -r "$REPO_DIR/requirements.txt" -q
echo "  Python packages installed"
echo ""

# --- Set up .env if it doesn't exist ---
echo "[4/6] Checking configuration..."
if [ ! -f "$REPO_DIR/config/.env" ]; then
    cp "$REPO_DIR/config/.env.example" "$REPO_DIR/config/.env"
    echo "  Created config/.env from template"
    echo "  >>> You MUST edit config/.env and add your GOOGLE_API_KEY <<<"
    echo "  >>> Get one free at: https://aistudio.google.com/apikey <<<"
else
    echo "  config/.env already exists"
fi
echo ""

# --- Verify critical dependencies ---
echo "[5/6] Verifying installation..."
ERRORS=0

# GStreamer
if command -v gst-launch-1.0 &> /dev/null; then
    GST_VERSION=$(gst-launch-1.0 --version | head -1 | awk '{print $NF}')
    echo "  GStreamer: $GST_VERSION -- OK"
else
    echo "  ERROR: GStreamer not installed. Camera will not work."
    ERRORS=$((ERRORS + 1))
fi

# OpenCV
if python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}', end='')" 2>/dev/null; then
    # Check GStreamer support in OpenCV
    if python3 -c "import cv2; assert 'GStreamer' in cv2.getBuildInformation()" 2>/dev/null; then
        echo " (with GStreamer) -- OK"
    else
        echo " (WARNING: no GStreamer support -- camera may not work)"
        echo "  The system OpenCV may lack GStreamer. If camera fails, rebuild OpenCV"
        echo "  with -DWITH_GSTREAMER=ON or use source='usb' in robot_config.yaml."
    fi
else
    echo "  ERROR: OpenCV not available"
    ERRORS=$((ERRORS + 1))
fi

# Gemini SDK
if python3 -c "import google.genai; print(f'  Gemini SDK: {google.genai.__version__} -- OK')" 2>/dev/null; then
    :
else
    echo "  ERROR: google-genai not installed"
    ERRORS=$((ERRORS + 1))
fi

# WebRTC (go2-webrtc-connect preferred, aiortc as fallback)
HAS_WEBRTC=0
if python3 -c "import go2_webrtc_driver; print('  go2-webrtc-connect -- OK')" 2>/dev/null; then
    HAS_WEBRTC=1
fi
if python3 -c "import aiortc; print(f'  aiortc: {aiortc.__version__} -- OK')" 2>/dev/null; then
    HAS_WEBRTC=1
fi
if [ "$HAS_WEBRTC" -eq 0 ]; then
    echo "  ERROR: No WebRTC library installed (need go2-webrtc-connect or aiortc)"
    ERRORS=$((ERRORS + 1))
fi

# CUDA (nice to have, not required)
if command -v nvcc &> /dev/null; then
    echo "  CUDA: $(nvcc --version | grep release | awk '{print $6}') -- OK"
else
    echo "  CUDA: not found (optional — AI runs in cloud, not locally)"
fi

echo ""

# --- Summary ---
echo "[6/6] Done!"
echo ""
echo "============================================"
if [ "$ERRORS" -gt 0 ]; then
    echo "  Setup completed with $ERRORS error(s)."
    echo "  Fix the errors above before running."
else
    echo "  Setup complete! No errors."
fi
echo ""
echo "  Next steps:"
echo "  1. Edit config/.env and add your GOOGLE_API_KEY"
echo "     nano $REPO_DIR/config/.env"
echo ""
echo "  2. Turn on the Go2 and connect the Jetson to its WiFi"
echo ""
echo "  3. Verify everything:"
echo "     source $VENV_DIR/bin/activate"
echo "     python $REPO_DIR/scripts/preflight_check.py"
echo ""
echo "  4. Run the robot:"
echo "     source $VENV_DIR/bin/activate"
echo "     python -m src.robot --interactive"
echo ""
echo "  IMPORTANT: Run 'source $VENV_DIR/bin/activate'"
echo "  every time you open a new terminal!"
echo "============================================"
