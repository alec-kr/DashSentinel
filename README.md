# DashSentinel
Real-time driver drowsiness detection system developed for deployment on edge devices.

Tested on Intel N100 and NVIDIA Jetson Orin Nano

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
