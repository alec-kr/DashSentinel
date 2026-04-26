# Sleepy Guard

A modular single-execution adaptive driver drowsiness monitor for Raspberry Pi and Linux PC.

## Features
- One execution script: `run_sleepy_guard.py`
- Works on Linux PC and Raspberry Pi
- Adaptive driver baseline saved across sessions
- Drowsiness scoring from EAR, blink rate, yawning, head pose, and closed-eye duration
- Optional buzzer alarm on Raspberry Pi GPIO
- Optional desktop UI or headless mode
- Optional CSV logging

## Structure
- `run_sleepy_guard.py` — single entrypoint
- `sleepy_guard/cli.py` — CLI arguments
- `sleepy_guard/app.py` — main runtime loop
- `sleepy_guard/features.py` — feature extraction and lighting enhancement
- `sleepy_guard/profile.py` — baseline persistence
- `sleepy_guard/scoring.py` — adaptive scoring
- `sleepy_guard/alarm.py` — desktop beep / Raspberry Pi buzzer
- `sleepy_guard/logging_utils.py` — CSV event logging
- `sleepy_guard/utils.py` — shared helpers
- `sleepy_guard/constants.py` — FaceMesh landmark constants

## Install
```bash
pip install -r requirements.txt
```

## Run on Linux PC
```bash
python3 run_sleepy_guard.py --show-ui --mirror
```

## Run on Raspberry Pi
```bash
python3 run_sleepy_guard.py --headless --enable-alarm --log-csv
```

## Optional flags
```bash
python3 run_sleepy_guard.py --show-ui --draw-landmarks --refine-landmarks
```

## Notes
- The same execution script works on both systems.
- `RPi.GPIO` is only needed if you want the buzzer alarm on Raspberry Pi.
- Data and logs are written to `./data` and `./logs` by default.


## startup baseline behavior
- on launch, the app first builds a short personal baseline from the current user's face and normal behavior
- it waits for enough clean frames with a neutral face and normal posture
- once that baseline is collected, it switches into normal sleepy / alert detection
- use `--rebuild-baseline-on-start` if you want to ignore the saved profile and rebuild from scratch

### example
```bash
python3 run_sleepy_guard.py --show-ui --mirror --rebuild-baseline-on-start
```


## ESP8266 USB serial telemetry

DashSentinel can send the live driver status, attentiveness score, and drowsy score to an ESP8266/ESP32 connected over USB.

Install dependency:

```bash
pip install pyserial
```

Run with serial enabled:

```bash
python3 run_sleepy_guard.py --show-ui --mirror --enable-esp-serial --esp-port /dev/ttyUSB0
```

Other common ports:

```bash
python3 run_sleepy_guard.py --show-ui --mirror --enable-esp-serial --esp-port /dev/ttyACM0
```

On Windows, use something like:

```bash
python run_sleepy_guard.py --show-ui --mirror --enable-esp-serial --esp-port COM3
```

Serial line format sent to the ESP:

```text
STATUS,ATTENTIVENESS,DROWSY_SCORE
```

Example:

```text
ALERT,87.4,0.123
```

An example Arduino sketch is included in:

```text
esp8266_display/DashSentinelSerialDisplay.ino
```
