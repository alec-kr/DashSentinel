# DashSentinel
Real-time driver drowsiness detection system developed for deployment on edge devices.

Tested devices:
- Intel N100
- NVIDIA Jetson Orin Nano
- Raspberry Pi 5

- Note:
MediaPipe-based landmark detection requires a 64-bit environment and is not supported on older Raspberry Pi devices (e.g., Pi 3/4 32-bit).

## Features
- Real-time face, eye, and behavior monitoring
- Adaptive user baseline calibration
- Drowsiness detection using:
  - Eye closure
  - Blink rate
  - Yawning detection
  - Head posture
- State machine (Alert, Warning, Drowsy)
- Works with USB cameras and DroidCam

## Tech Stack
- Python
- OpenCV
- NumPy

## Usage
```bash
python3 run_sleepy_guard.py --show-ui --mirror --rebuild-baseline-on-start
```
