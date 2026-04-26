# sleepy guard app
# real-time driver monitoring using face landmarks and adaptive scoring

import json
import os

import numpy as np

from .utils import ensure_parent


# class that groups related logic/state for this component
class RunningStat:
# initialize class state and configuration
    def __init__(self, mean=0.0, var=1.0, count=0.0):
        self.mean = float(mean)
        self.m2 = float(max(var, 1e-6) * max(count, 1.0))
        self.count = float(count)

# update running statistics / profile with new data
    def update(self, x: float):
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
# function that handles a specific step in the pipeline
    def var(self):
        if self.count <= 1:
            return 1e-4
        return max(self.m2 / (self.count - 1.0), 1e-4)

    @property
# function that handles a specific step in the pipeline
    def std(self):
        return float(np.sqrt(self.var))

# function that handles a specific step in the pipeline
    def to_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    @classmethod
# function that handles a specific step in the pipeline
    def from_dict(cls, data, default_mean, default_var):
        if not data:
            return cls(default_mean, default_var, 0.0)
        return cls(data.get("mean", default_mean), data.get("var", default_var), data.get("count", 0.0))


# class that groups related logic/state for this component
class DriverProfile:
    DEFAULTS = {
        "ear": (0.28, 0.0025),
        "mar": (0.16, 0.0020),
        "blink_rate": (15.0, 16.0),
        "roll_deg": (0.0, 20.0),
        "yaw_ratio": (0.0, 0.01),
        "pitch_ratio": (0.58, 0.01),
    }

# initialize class state and configuration
    def __init__(self, path: str):
        self.path = path
        self.stats = {}
        self.sessions = 0
        self.total_updates = 0
        self._load()
    
    # save profile to disk
    def save(self):
        import json, os
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "sessions": self.sessions,
                "total_updates": self.total_updates,
                "stats": {k: v.to_dict() for k, v in self.stats.items()}
            }, f, indent=2)

# load data from disk (profile or config)
    def _load(self):
        payload = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

        self.sessions = int(payload.get("sessions", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        saved = payload.get("stats", {})

        for key, (default_mean, default_var) in self.DEFAULTS.items():
            self.stats[key] = RunningStat.from_dict(saved.get(key), default_mean, default_var)

    # reset baseline completely (for new user / fresh calibration)
    def reset_baseline(self):
        self.total_updates = 0
        self.stats = {}

        for key, (mean, var) in self.DEFAULTS.items():
            self.stats[key] = RunningStat(mean, var, 0)
    
    # function that handles a specific step in the pipeline
    def mean(self, key: str) -> float:
        return self.stats[key].mean

# function that handles a specific step in the pipeline
    def std(self, key: str) -> float:
        return self.stats[key].std

# update running statistics / profile with new data
    def update_from_alert_frame(self, features: dict):
        for key in self.DEFAULTS.keys():
            self.stats[key].update(float(features[key]))
        self.total_updates += 1

# function that handles a specific step in the pipeline
    def begin_session(self):
        self.sessions += 1
