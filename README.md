# DashSentinel

----Work in progress on README----

Real-time adaptive driver monitoring system.

DashSentinel is a modular driver attentiveness system that uses computer vision to detect drowsiness and outputs real-time status to an embedded device (ESP8266/ESP32) via USB.

## Features
- One execution script: `run_dashsentinel.py`
- Adaptive driver baseline saved across sessions
- Drowsiness scoring from eye aspect ratio (EAR), blink rate, yawning, head pose, and closed-eye duration
- Can run a desktop UI or headless mode

## Project Structure
- `run_dashsentinel.py` — single entrypoint for execution
- `DashSentinel/cli.py` — CLI arguments
- `DashSentinel/app.py` — main runtime loop
- `DashSentinel/features.py` — feature extraction and lighting enhancement
- `DashSentinel/profile.py` — baseline persistence
- `DashSentinel/scoring.py` — adaptive scoring
- `DashSentinel/logging_utils.py` — CSV event logging
- `DashSentinel/utils.py` — some shared helpers
- `DashSentinel/constants.py` — FaceMesh landmark constants

## Install
```bash
pip install -r requirements.txt
```

## Run with UI
```bash
python3 run_dashsentinel.py --show-ui --mirror
```

## Run headless
```bash
python3 run_dashsentinel.py --headless --log-csv
```

## Optional flags
```bash
python3 run_dashsentinel.py --show-ui --draw-landmarks --refine-landmarks
```

## Notes
- Data and logs are written to `./data` and `./logs` by default.


## startup baseline behavior
- on launch, the app builds a personal baseline from the current user's face and normal behavior
- it waits for enough clean frames with a neutral face and normal posture
- once that baseline is collected, it switches into normal sleepy / alert detection
- use `--rebuild-baseline-on-start` if you want to ignore the saved profile and rebuild from scratch

### example
```bash
python3 run_dashsentinel.py --show-ui --mirror --rebuild-baseline-on-start
```


## ESP8266 USB serial telemetry

DashSentinel can send the live driver status, attentiveness score, and drowsy score to an ESP8266/ESP32 connected over USB.

Install dependency:

```bash
pip install pyserial
```

Run with serial enabled and select your port with ```--esp-port```:

```bash
python3 run_dashsentinel.py --show-ui --mirror --enable-esp-serial --esp-port /dev/ttyUSB0
```


Serial line format sent to the ESP:

```text
STATUS,ATTENTIVENESS,DROWSY_SCORE
```

Example:

```text
ALERT,87.4,0.123
```