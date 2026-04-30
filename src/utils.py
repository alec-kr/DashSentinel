import os
import platform
import time
from pathlib import Path

import numpy as np


def clamp(value, low, high):
    return max(low, min(high, value))

def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

# ensures that the parent directory of the given path exists, creating it if necessary
def ensure_parent(path: str):
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

# get current timestamp
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
