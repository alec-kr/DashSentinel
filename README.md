# 🚗 DashSentinel

Real-time Driver Drowsiness Detection System with Embedded Feedback Interface

## Overview

The system combines:

- Computer Vision (MediaPipe / OpenCV) for facial feature extraction
- Machine Learning / Heuristics for drowsiness classification
- Embedded System (ESP8266) for real-time display and user interaction

It is designed as a prototype for a deployable driver safety product in fleets/personal use.

## Features
- Real-time face tracking and feature extraction
- Driver attentiveness scoring (%)
- Drowsiness detection based on:
    - eye closure patterns
    - yawning detection
    - attentiveness %

- OLED display output:
    - status (ALERT / WARNING / DROWSY)

- LED indicators:
    - 🟢 ALERT → solid green
    - 🟡 WARNING → blinking yellow
    - 🔴 DROWSY → fast blinking red

- Physical buttons:
    - Reset baseline (user-specific calibration)
    - Reset stats (data stored over-time)

## Tech Stack
### 💾 Software
- Python 3
- OpenCV
- MediaPipe
- TensorFlow (optional for model extensions)
- PySerial

### ⚙️ Hardware
- ESP8266 (NodeMCU)
- SSD1306 OLED (I2C, 128x64)
- Push buttons (x2)
- LEDs (red, yellow, green)
- Resistors (3x 330 ohms)

## 📁 Project Structure

```text

DashSentinel/
├── run_dashsentinel.py        # main entrypoint for execution
│
├── src/                       # core application logic
│   ├── app.py                 # main app orchestration
│   ├── features.py            # feature extraction (ear, yawning, etc.)
│   ├── model.py               # scoring / detection logic
│   └── serial.py              # esp8266 communication layer
│
├── DisplayModule/             # esp8266 firmware (platformio project)
│   └── src/
│       └── main.cpp           # oled, leds, buttons logic
│
├── nodemcu_carrier_pcb/       # hardware design (KiCad)
│
├── data/                      # runtime-generated data
│   └── driver_profile.json    # user-specific baseline + stats
│
├── requirements.txt           # python dependencies
└── README.md                  # project documentation

```

## Install
```bash
pip install -r requirements.txt
```

## Run headless
```bash
python3 run_dashsentinel.py --headless --log-csv
```

## Optional flags
```bash
python3 run_dashsentinel.py --show-ui --draw-landmarks --refine-landmarks
```

## Startup Baseline
- on launch, the app builds a personal baseline from the current user's face and normal behavior
- it waits for enough clean frames with a neutral face and normal posture
- once that baseline is collected, it switches into normal sleepy / alert detection
- use `--rebuild-baseline-on-start` if you want to ignore the saved profile and rebuild from scratch

### Example
```bash
python3 run_dashsentinel.py --show-ui --mirror --rebuild-baseline-on-start
```

## 🚀 Future Improvements
- Replace heuristic model with trained ML model
- Mobile app integration
- Full standalone embedded unit (dashcam form factor)