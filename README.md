# 🚗 DashSentinel

**Real-time Driver Drowsiness Detection System with Embedded Feedback Interface**

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
|
├── schematic/                 # schematic for all connections
│
├── data/                      # runtime-generated data
│   └── driver_profile.json    # user-specific baseline + stats
|
├── media/                     # Images showing the system in operation
│
├── requirements.txt           # python dependencies
└── README.md                  # project documentation

```

## Install requirements
```bash
pip install -r requirements.txt
```


## ▶️ Run the Project

### With UI and ESP8266 Interface
```bash
python3 run_dashsentinel.py --show-ui --draw-landmarks --enable-esp-serial --esp-port /dev/ttyUSB0
```

### Headless mode (logging enabled)
```bash
python3 run_dashsentinel.py --headless --log-csv --enable-esp-serial --esp-port /dev/ttyUSB0
```

### Optional flags (see ```src/cli.py``` for full list of options)
```bash
python3 run_dashsentinel.py --show-ui --draw-landmarks --refine-landmarks
```

## Startup Calibration
At startup:
- system learns user-specific facial metrics
- establishes baseline EAR and behavior

## Reset options
- Hardware button → reset baseline
- CLI flag → rebuild baseline at launch

```bash
python3 run_dashsentinel.py --show-ui --mirror --rebuild-baseline-on-start
```

## 🚀 Future Improvements
- Replace heuristic model with trained ML model
- Mobile app integration
- Full standalone embedded unit (dashcam form factor)