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
4. **WiFi router with internet** — Starlink, home router, hotspot, anything
5. **Gemini API key** — free at https://aistudio.google.com/apikey

## Networking — Read This First

Everything connects over **one WiFi network**. No Ethernet cable needed.

```
                    ┌──────────────┐
                    │   Starlink   │
                    │  (or any     │
                    │   router)    │
                    └──────┬───────┘
                           │ WiFi
             ┌─────────────┼─────────────┐
             │             │             │
        ┌────┴────┐  ┌─────┴─────┐  ┌───┴────┐
        │  Go2    │  │  Jetson   │  │ Phone  │
        │  Pro    │  │  Orin     │  │(setup  │
        │ (robot) │  │ (brain)   │  │  only) │
        └─────────┘  └───────────┘  └────────┘
```

**How it works:** The Go2 joins your WiFi in "STA-L mode" (station mode). The Jetson
connects to the same WiFi. They talk to each other over the local network, and the
Jetson reaches the Gemini API through the same connection. One network does it all.

**Do I need Ethernet?** No. Ethernet is only needed if you use "AP mode" (where the
Go2 creates its own hotspot and hogs the Jetson's WiFi). If you have any WiFi router
with internet (Starlink, home router, phone hotspot), use STA-L mode instead and
skip Ethernet entirely.

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

### Step 3: Connect everything to your WiFi

You need three things on the same WiFi network: the Go2, the Jetson, and your phone (for initial setup).

#### 3a. Put the Go2 on your WiFi (STA-L mode)

The Go2 ships in "AP mode" (it creates its own hotspot). You need to switch it to
"STA-L mode" so it joins your existing WiFi instead.

1. **Turn on the Go2** and wait for it to boot (~30 seconds, you'll hear a chime)
2. **Open the Unitree app** on your Android phone
3. **Connect to the Go2's temporary hotspot:**
   - Your phone will see a WiFi network named `GO2_XXXX` — connect to it
   - Open the Unitree app and pair with the dog
4. **Switch the Go2 to STA-L mode:**
   - In the app: **Settings → Networking → Station Mode (STA-L)**
   - Select your WiFi network (e.g., your Starlink network)
   - Enter the WiFi password
   - The Go2 will reboot and join your network
5. **Find the Go2's new IP address:**
   - Check the Unitree app — it usually shows the IP after reconnecting
   - Or check your router's admin page / DHCP client list
   - The IP will look something like `192.168.1.210` (depends on your router)
6. **Reconnect your phone to your normal WiFi** (it's still on the Go2's old hotspot)
7. **Re-pair the app** — open Unitree app, it should find the Go2 on your network

#### 3b. Connect the Jetson to the same WiFi

```bash
# List available networks
nmcli device wifi list

# Connect (use your actual network name and password)
nmcli device wifi connect "YourStarlinkNetwork" password "your-password"

# Verify you have internet
curl -s https://generativelanguage.googleapis.com/ > /dev/null && echo "OK" || echo "NO INTERNET"
```

#### 3c. Verify the Jetson can reach the Go2

```bash
# Use the Go2's IP from step 3a
ping 192.168.1.210
```

You should see replies. If not, double-check both devices are on the same network.

#### 3d. Update the config

Edit `config/robot_config.yaml`:

```yaml
unitree:
  connection_method: "sta"        # STA-L mode — everything on one WiFi
  robot_ip: "192.168.1.210"       # <-- replace with your Go2's actual IP
```

That's it. No Ethernet. No second WiFi adapter. One network.

### Step 4: Verify everything works

```bash
source venv/bin/activate
python scripts/preflight_check.py
```

This checks your API key, Go2 connection, camera, and all dependencies. Fix anything it flags before continuing.

### Step 5: Run it!

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

## Alternative Network Setups

Most people should use STA-L mode above. These are for special cases only.

<details>
<summary><strong>AP Mode — No WiFi router available</strong></summary>

The Go2 creates its own WiFi hotspot. The Jetson connects to it directly.

**Downside:** The Jetson's WiFi is used up talking to the Go2, so you need a
**second network connection** (Ethernet cable or USB WiFi adapter) for internet.

1. Turn on the Go2 — it creates a `GO2_XXXX` hotspot
2. Connect Jetson to the hotspot:
   ```bash
   nmcli device wifi connect "GO2_XXXX" password "the-password"
   ```
3. Plug **Ethernet** into the Jetson for internet (Starlink router, any switch, etc.)
4. Verify:
   ```bash
   ping 192.168.12.1              # Should reach the Go2
   curl https://google.com        # Should have internet via Ethernet
   ```
5. Config stays at defaults:
   ```yaml
   unitree:
     connection_method: "ap"
     robot_ip: "192.168.12.1"
   jetson:
     robot_interface: "wlan0"     # WiFi talks to Go2
     internet_interface: "eth0"   # Ethernet talks to internet
   ```

</details>

<details>
<summary><strong>Ethernet — Direct cable to Go2 (lowest latency)</strong></summary>

For EDU models or advanced setups. Plug Ethernet directly between Jetson and Go2.

1. Connect Ethernet cable between Jetson and Go2
2. Set a static IP:
   ```bash
   sudo nmcli con add type ethernet con-name go2 ifname eth0 ip4 192.168.123.99/24
   ```
3. Use Jetson's WiFi for internet
4. Verify: `ping 192.168.123.18`

</details>

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

### Can't ping the Go2
- Is the Go2 powered on? (check the lights)
- Are both the Jetson and Go2 on the **same WiFi network**?
  ```bash
  # What network is the Jetson on?
  nmcli -t -f active,ssid dev wifi | grep "^yes"
  ```
- Did you switch the Go2 to STA-L mode? (Step 3a) If it's still in AP mode, the Go2 is on its own hotspot, not your WiFi
- Try rebooting the Go2 — hold the power button for 3 seconds, wait, power on again
- If using AP mode with Ethernet, make sure `config/robot_config.yaml` has the right `robot_interface` (see AP Mode section)

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

### Ping works but WebRTC connection fails
- Make sure no firewall is blocking UDP traffic between the Jetson and Go2
- On Starlink/mesh networks: check that "client isolation" or "AP isolation" is disabled in your router settings — this blocks devices from talking to each other even though they're on the same network

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
