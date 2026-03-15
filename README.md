# Gemini Robot Dog

AI-powered robot dog using Google Gemini for vision and reasoning. Tell it what to do in plain English and it figures out the rest.

**Works with:** Unitree Go2 Pro / Pro 2 / EDU + NVIDIA Jetson Orin Nano / NX / AGX

```
You: "walk to the red cone"
Gemini: sees the cone through the camera, plans a path, walks there
```

## What You Need

1. **Unitree Go2 Pro** (or Pro 2 or EDU) — the robot dog
2. **NVIDIA Jetson Orin Nano** (or NX/AGX) — the brain, mounted on or near the dog
3. **Android phone** — for initial Go2 setup via the Unitree app
4. **Gemini API key** — free at https://aistudio.google.com/apikey
5. **WiFi** — the Jetson needs internet for Gemini API calls

## Setup (15 minutes)

> Your Jetson should already have JetPack 6.x flashed (Ubuntu 22.04 + CUDA).
> If not, flash it first: https://developer.nvidia.com/jetpack-sdk-62

### Step 1: Clone and install

SSH into your Jetson (or open a terminal on it) and run:

```bash
git clone <this-repo>
cd ROBOT
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

This takes ~10 minutes. It installs GStreamer, OpenCV, and all Python dependencies.

### Step 2: Add your API key

```bash
cp config/.env.example config/.env
nano config/.env
```

Paste your Gemini API key on the `GOOGLE_API_KEY=` line:

```
GOOGLE_API_KEY=AIzaSy...your-key-here
```

Save and exit (Ctrl+X, then Y, then Enter).

### Step 3: Connect to the Go2

Pick **one** of these three options:

**Option A — AP Mode (default, no existing WiFi needed):**

The Go2 creates its own WiFi hotspot. Simple but requires a second network adapter on the Jetson for internet.

1. Turn on the Go2 — it creates a WiFi hotspot
2. Open the **Unitree app** on your Android phone and connect to the dog
3. Note the WiFi name and password from the app
4. Connect the Jetson to the Go2's WiFi:
   ```bash
   nmcli device wifi connect "GO2_XXXX" password "the-password"
   ```
5. Verify: `ping 192.168.12.1` — you should get replies

Config: `connection_method: "ap"` and `robot_ip: "192.168.12.1"` (these are the defaults).

**Option B — STA-L Mode (recommended if you have a WiFi router):**

Both the Go2 and Jetson join your existing WiFi network. This is the easiest setup because the Jetson gets internet and robot access over one connection — no second adapter needed.

1. Use the **Unitree app** to put the Go2 in STA-L mode:
   App → Settings → Networking → Station Mode → select your WiFi network
2. The Go2 will get an IP from your router (e.g., `192.168.1.210`)
   Check the app or your router's DHCP table for the assigned IP
3. Connect the Jetson to the **same WiFi network**
4. Verify: `ping 192.168.1.210` (use your dog's actual IP)
5. Edit `config/robot_config.yaml`:
   ```yaml
   unitree:
     connection_method: "sta"
     robot_ip: "192.168.1.210"   # your dog's IP on the shared network
   ```

**Option C — Ethernet (lowest latency, for EDU or advanced setups):**

1. Plug an Ethernet cable between the Jetson and Go2
2. Set a static IP:
   ```bash
   sudo nmcli con add type ethernet con-name go2 ifname eth0 ip4 192.168.123.99/24
   ```
3. Verify: `ping 192.168.123.18`

### Step 4: Give the Jetson internet access

The Jetson needs internet for Gemini API calls.

- **If using STA-L mode (Option B):** You're already done — the Jetson has internet through the shared WiFi network.
- **If using AP mode (Option A):** Plug in a **USB WiFi adapter** or Ethernet cable for internet, since the built-in WiFi is connected to the Go2's hotspot.
- **If using Ethernet (Option C):** Use the Jetson's built-in WiFi for internet.

Verify internet works: `curl -s https://generativelanguage.googleapis.com/ > /dev/null && echo "OK" || echo "NO INTERNET"`

### Step 5: Verify everything works

```bash
source venv/bin/activate
python scripts/preflight_check.py
```

This checks your API key, Go2 connection, camera, and all dependencies. Fix anything it flags before continuing.

### Step 6: Run it!

```bash
source venv/bin/activate

# Interactive mode — type commands to your robot
python -m src.robot --interactive

# Single instruction
python -m src.robot --instruction "walk forward to the red cone"

# Continuous loop (robot keeps executing the instruction)
python -m src.robot --instruction "follow the person" --continuous

# With web dashboard (opens a browser UI on port 8080)
python -m src.robot --interactive --web
```

**Every time you open a new terminal**, run `source venv/bin/activate` first.

## Web Dashboard

Add `--web` to any run mode to start a local browser UI:

```bash
python -m src.robot --interactive --web
# Open http://<jetson-ip>:8080 in your browser
```

The dashboard provides:
- **Emergency stop button** — big red button, also triggered by spacebar
- **Live status** — battery, connection, IMU, camera FPS
- **Manual controls** — WASD / arrow keys or on-screen D-pad
- **Natural language input** — type instructions for Gemini
- **Camera feed** — live MJPEG stream of what the robot sees
- **Robot ping** — one-click connectivity check
- **API key setup** — enter your Gemini key without SSH
- **Log viewer** — live tail of `logs/robot.log`

Access it from any device on the same network (phone, tablet, laptop). Use `--web-port 9090` to change the port.

## Kill Switch (Emergency Stop)

| How | What happens |
|-----|-------------|
| **Ctrl+C** (once) | Graceful stop: robot sits down, disconnects |
| **Ctrl+C** (twice) | EMERGENCY: motors go limp immediately |
| **Spacebar** | EMERGENCY: same as double Ctrl+C (non-interactive modes) |
| **Hold Go2 power button** (3 sec) | Hardware shutdown — last resort |

## How It Works

```
Go2 front camera --> Jetson Orin --> Gemini API (cloud) --> Jetson --> Go2 motors
      (H264)        (captures       (sees image,           (sends    (walks,
                     frame)          plans actions)          commands)  turns)
```

1. The Go2's built-in front camera streams video to the Jetson over WiFi
2. The Jetson captures a frame and sends it to Gemini with your instruction
3. Gemini analyzes the scene and returns a plan (e.g., "turn left 30 degrees, walk forward 2 meters")
4. The Jetson sends movement commands to the Go2 via WebRTC
5. Repeat

## Go2 Pro 2 Notes

The Go2 Pro 2 works the same as the Pro, with one difference: it may require a **WebRTC token** for communication. To set it up:

1. Open the Unitree app on your phone
2. Go to Settings > Robot > Token
3. Copy the token
4. Add it to your `config/.env`:
   ```
   UNITREE_TOKEN=your-token-here
   ```

If connection works without a token, you can skip this.

## Troubleshooting

### "ModuleNotFoundError: No module named ..."
You forgot to activate the virtual environment:
```bash
source venv/bin/activate
```

### "GOOGLE_API_KEY environment variable is required"
You didn't set up your API key. Go back to Step 2.

### "ping 192.168.12.1" doesn't work
- Is the Go2 powered on? (check the lights)
- Is the Jetson connected to the Go2's WiFi? Run `nmcli device wifi list` to see available networks
- Try rebooting the Go2

### "WebRTC signaling failed (HTTP 403)"
Your Go2 Pro 2 needs a token. See "Go2 Pro 2 Notes" above.

### "Failed to open camera"
- The Go2 must be on and connected — the camera streams over the network
- Test the stream directly:
  ```bash
  gst-launch-1.0 udpsrc address=230.1.1.1 port=1720 ! application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
  ```
  If this doesn't show video, the Go2 isn't streaming (check WiFi connection)

### "No camera frame available" (robot doesn't move)
The camera connected but no frames arrived yet. Wait a few seconds. If it persists, check your WiFi signal strength — move the Jetson closer to the Go2.

### Robot does nothing / Gemini returns empty plans
- Check internet: `curl https://generativelanguage.googleapis.com/`
- Check API key: Make sure there are no extra spaces in your `.env` file
- Try a simpler instruction: "stand up" or "walk forward"

### Robot moves erratically
- Reduce speed: edit `config/robot_config.yaml`, set `max_linear_velocity: 0.3`
- Make sure the Go2 is on flat ground with space to move
- Check battery: low battery causes erratic behavior

## Project Structure

```
ROBOT/
├── config/
│   ├── robot_config.yaml    # Camera, network, speed settings
│   └── .env.example         # API key template
├── scripts/
│   ├── setup_jetson.sh      # Installs everything
│   └── preflight_check.py   # Verifies your setup
├── src/
│   ├── robot.py             # Main controller (start here)
│   ├── gemini/
│   │   └── robotics_client.py   # Talks to Gemini API
│   ├── unitree/
│   │   └── go2_controller.py    # Talks to Go2 via WebRTC
│   ├── vision/
│   │   └── camera.py            # Captures camera frames
│   └── web/
│       ├── server.py            # Local web dashboard (--web flag)
│       └── templates/
│           └── index.html       # Dashboard UI
└── README.md
```

## Supported Hardware

| Robot | Status | Notes |
|-------|--------|-------|
| Go2 Pro | Fully supported | Needs external Jetson mounted on/near robot |
| Go2 Pro 2 | Supported | May need WebRTC token (see above) |
| Go2 EDU | Supported | Has built-in Jetson; can also use CycloneDDS |
| Go2 AIR | Should work | Untested, same WebRTC protocol |

| Jetson | Status | Notes |
|--------|--------|-------|
| Orin Nano (8GB) | Supported | All AI runs in cloud; 8GB is fine |
| Orin NX | Supported | |
| Orin AGX | Supported | |

## API Access

This project uses **Gemini Robotics-ER 1.5** which is publicly available via the Gemini API. No waitlist needed. Get your key at https://aistudio.google.com/apikey
