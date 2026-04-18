# sleepy guard app
# real-time driver monitoring using face landmarks and adaptive scoring

import time
from collections import deque
from dataclasses import dataclass

from .utils import clamp


@dataclass
# class that groups related logic/state for this component
class ScoreOutput:
    phase: str
    status: str
    confidence: float
    drowsy_score: float
    attentiveness: float
    should_alarm: bool


# class that groups related logic/state for this component
class AdaptiveScorer:
# initialize class state and configuration
    def __init__(self, profile, calibration_seconds=25, alarm_threshold=0.72, attention_window=240):
        self.profile = profile
        self.calibration_seconds = calibration_seconds
        self.alarm_threshold = alarm_threshold
# current timestamp used for fps or timing logic
        self.start_time = time.time()
        self.attention_history = deque(maxlen=attention_window)

# function that handles a specific step in the pipeline
    def seconds_since_start(self):
# current timestamp used for fps or timing logic
        return time.time() - self.start_time

# function that handles a specific step in the pipeline
    def in_calibration(self):
        return self.seconds_since_start() < self.calibration_seconds or self.profile.total_updates < 150

# function that handles a specific step in the pipeline
    def _normalized_delta(self, value, mean, std, floor):
        return clamp(abs(value - mean) / max(2.5 * std, floor), 0.0, 1.0)

# combine features into a drowsiness score and status
    def score(self, features: dict) -> ScoreOutput:
# compute eye aspect ratio to detect eye closure
        ear_drop = clamp((self.profile.mean("ear") - features["ear"]) / max(2.2 * self.profile.std("ear"), 0.03), 0.0, 1.0)
        low_blink = clamp((self.profile.mean("blink_rate") - features["blink_rate"]) / max(2.0 * self.profile.std("blink_rate"), 4.0), 0.0, 1.0)
        yawn_mag = clamp((features["mar"] - max(self.profile.mean("mar") + 2.0 * self.profile.std("mar"), 0.34)) / 0.22, 0.0, 1.0)
        roll_delta = self._normalized_delta(features["roll_deg"], self.profile.mean("roll_deg"), self.profile.std("roll_deg"), 12.0)
        yaw_delta = self._normalized_delta(features["yaw_ratio"], self.profile.mean("yaw_ratio"), self.profile.std("yaw_ratio"), 0.08)
        pitch_delta = self._normalized_delta(features["pitch_ratio"], self.profile.mean("pitch_ratio"), self.profile.std("pitch_ratio"), 0.12)

        drowsy_score = (
            0.30 * ear_drop
            + 0.18 * features["closed_frames_norm"]
            + 0.10 * low_blink
            + 0.14 * yawn_mag
            + 0.10 * features["yawn_flag"]
            + 0.08 * roll_delta
            + 0.05 * yaw_delta
            + 0.05 * pitch_delta
        )
        drowsy_score = clamp(drowsy_score, 0.0, 1.0)

        if self.in_calibration():
            phase = "CALIBRATING"
            status = "LEARNING BASELINE"
            confidence = clamp(0.50 + 0.50 * (self.seconds_since_start() / max(self.calibration_seconds, 1)), 0.0, 1.0)
        else:
            phase = "ACTIVE"
            status = "DROWSY" if drowsy_score >= 0.5 else "ALERT"
            confidence = drowsy_score if status == "DROWSY" else (1.0 - drowsy_score)

        attentiveness = (1.0 - drowsy_score) * 100.0
        self.attention_history.append(attentiveness)
        attentiveness_smoothed = sum(self.attention_history) / len(self.attention_history)

        stable_alert = (
            drowsy_score < 0.28
            and features["yawn_flag"] < 0.5
            and features["posture_flag"] < 0.5
            and features["closed_frames_norm"] < 0.2
        )
        if self.in_calibration() or stable_alert:
            self.profile.update_from_alert_frame(features)

        should_alarm = (not self.in_calibration()) and drowsy_score >= self.alarm_threshold
        return ScoreOutput(
            phase=phase,
            status=status,
            confidence=float(confidence),
            drowsy_score=float(drowsy_score),
            attentiveness=float(attentiveness_smoothed),
            should_alarm=should_alarm,
        )
