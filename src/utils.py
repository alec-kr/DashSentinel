# sleepy guard app
# real-time driver monitoring using face landmarks and adaptive scoring

import os
import platform
import time
from pathlib import Path

import numpy as np


# function that handles a specific step in the pipeline
def clamp(value, low, high):
    return max(low, min(high, value))


# function that handles a specific step in the pipeline
def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


# function that handles a specific step in the pipeline
def ensure_parent(path: str):
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


# function that handles a specific step in the pipeline
def is_raspberry_pi() -> bool:
    try:
        if platform.system().lower() != "linux":
            return False
        model_path = "/sys/firmware/devicetree/base/model"
        if os.path.exists(model_path):
            with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
                return "raspberry pi" in f.read().lower()
    except Exception:
        pass
    return False


# function that handles a specific step in the pipeline
def beep_desktop():
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass


# function that handles a specific step in the pipeline
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
