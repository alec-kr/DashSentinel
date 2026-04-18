# sleepy guard app
# real-time driver monitoring using face landmarks and adaptive scoring

import argparse


# function that handles a specific step in the pipeline
def parse_args():
    parser = argparse.ArgumentParser(description="Single-executable adaptive sleepy driver monitor for Raspberry Pi and Linux PC.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-fps", type=int, default=20)
    parser.add_argument("--process-every-n-frames", type=int, default=1)
    parser.add_argument("--mirror", action="store_true")

    parser.add_argument("--profile-path", type=str, default="./data/driver_profile.json")
    parser.add_argument("--log-path", type=str, default="./logs/sleepy_guard_events.csv")
    parser.add_argument("--log-csv", action="store_true")

    parser.add_argument("--calibration-seconds", type=int, default=25)
    parser.add_argument("--ear-threshold", type=float, default=0.23)
    parser.add_argument("--yawn-mar-threshold", type=float, default=0.45)
    parser.add_argument("--yawn-frames-threshold", type=int, default=12)
    parser.add_argument("--alarm-threshold", type=float, default=0.72)
    parser.add_argument("--attention-window", type=int, default=240)

    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--refine-landmarks", action="store_true")
    parser.add_argument("--draw-landmarks", action="store_true")

    parser.add_argument("--enable-alarm", action="store_true")
    parser.add_argument("--buzzer-pin", type=int, default=18)
    parser.add_argument("--alarm-cooldown-seconds", type=float, default=6.0)
    parser.add_argument("--alarm-duration-seconds", type=float, default=0.15)

    parser.add_argument("--show-ui", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--window-name", type=str, default="Sleepy Guard")
    parser.add_argument("--save-profile-every-seconds", type=int, default=15)
    return parser.parse_args()
