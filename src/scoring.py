import time
from collections import deque
from dataclasses import dataclass

from .utils import clamp


@dataclass
class ScoreOutput:
    phase: str
    status: str
    confidence: float
    drowsy_score: float
    attentiveness: float
    should_alarm: bool
    reason: str


class AdaptiveScorer:
    def __init__(
        self,
        profile,
        calibration_seconds=25,
        alarm_threshold=0.72,
        attention_window=240,
        warning_threshold=0.42,
        drowsy_threshold=0.58,
        status_hold_frames=12,
        no_face_hold_frames=20,
    ):
        self.profile = profile
        self.calibration_seconds = calibration_seconds
        self.alarm_threshold = alarm_threshold
        self.warning_threshold = warning_threshold
        self.drowsy_threshold = drowsy_threshold
        self.status_hold_frames = status_hold_frames
        self.no_face_hold_frames = no_face_hold_frames

        self.start_time = time.time()
        self.attention_history = deque(maxlen=attention_window)
        self.score_history = deque(maxlen=30)
        self.state = "CALIBRATING"
        self.pending_state = None
        self.pending_frames = 0
        self.no_face_frames = 0

    def seconds_since_start(self):
        return time.time() - self.start_time

    def in_calibration(self):
        return self.seconds_since_start() < self.calibration_seconds or self.profile.total_updates < 150

    def _normalized_delta(self, value, mean, std, floor):
        return clamp(abs(value - mean) / max(2.5 * std, floor), 0.0, 1.0)

    def _build_reason(self, components):
        ranked = sorted(components.items(), key=lambda item: item[1], reverse=True)
        top = [name for name, score in ranked if score >= 0.18][:2]
        if not top:
            return "normal behavior"
        return ", ".join(top)

    def _promote_state(self, desired_state):
        if desired_state == self.state:
            self.pending_state = None
            self.pending_frames = 0
            return

        if self.pending_state != desired_state:
            self.pending_state = desired_state
            self.pending_frames = 1
            return

        self.pending_frames += 1
        if self.pending_frames >= self.status_hold_frames:
            self.state = desired_state
            self.pending_state = None
            self.pending_frames = 0

    def update_no_face(self):
        self.no_face_frames += 1
        if self.no_face_frames >= self.no_face_hold_frames:
            self.state = "NO_FACE"
            self.pending_state = None
            self.pending_frames = 0

        attentiveness = sum(self.attention_history) / len(self.attention_history) if self.attention_history else 100.0
        return ScoreOutput(
            phase="ACTIVE" if not self.in_calibration() else "CALIBRATING",
            status="NO FACE",
            confidence=0.0,
            drowsy_score=self.score_history[-1] if self.score_history else 0.0,
            attentiveness=float(attentiveness),
            should_alarm=False,
            reason="face not visible",
        )

    def score(self, features: dict) -> ScoreOutput:
        self.no_face_frames = 0

        ear_drop = clamp((self.profile.mean("ear") - features["ear"]) / max(2.2 * self.profile.std("ear"), 0.03), 0.0, 1.0)
        low_blink = clamp((self.profile.mean("blink_rate") - features["blink_rate"]) / max(2.0 * self.profile.std("blink_rate"), 4.0), 0.0, 1.0)
        yawn_mag = clamp((features["mar"] - max(self.profile.mean("mar") + 2.0 * self.profile.std("mar"), 0.34)) / 0.22, 0.0, 1.0)
        roll_delta = self._normalized_delta(features["roll_deg"], self.profile.mean("roll_deg"), self.profile.std("roll_deg"), 12.0)
        yaw_delta = self._normalized_delta(features["yaw_ratio"], self.profile.mean("yaw_ratio"), self.profile.std("yaw_ratio"), 0.08)
        pitch_delta = self._normalized_delta(features["pitch_ratio"], self.profile.mean("pitch_ratio"), self.profile.std("pitch_ratio"), 0.12)

        components = {
            "eye closure": 0.30 * ear_drop + 0.18 * features["closed_frames_norm"],
            "low blink rate": 0.10 * low_blink,
            "yawning": 0.14 * yawn_mag + 0.10 * features["yawn_flag"],
            "head tilt": 0.08 * roll_delta + 0.05 * yaw_delta + 0.05 * pitch_delta,
        }

        drowsy_score = sum(components.values())
        drowsy_score = clamp(drowsy_score, 0.0, 1.0)
        self.score_history.append(drowsy_score)
        smoothed_score = sum(self.score_history) / len(self.score_history)

        stable_alert = (
            smoothed_score < 0.28
            and features["yawn_flag"] < 0.5
            and features["posture_flag"] < 0.5
            and features["closed_frames_norm"] < 0.2
        )
        if self.in_calibration() or stable_alert:
            self.profile.update_from_alert_frame(features)

        if self.in_calibration():
            self.state = "CALIBRATING"
            confidence = clamp(0.50 + 0.50 * (self.seconds_since_start() / max(self.calibration_seconds, 1)), 0.0, 1.0)
            reason = "learning normal baseline"
        else:
            if smoothed_score >= self.drowsy_threshold:
                desired_state = "DROWSY"
            elif smoothed_score >= self.warning_threshold:
                desired_state = "WARNING"
            else:
                desired_state = "ALERT"

            if self.state == "NO_FACE":
                self.state = "ALERT"
            self._promote_state(desired_state)
            confidence = smoothed_score if self.state in ("WARNING", "DROWSY") else (1.0 - smoothed_score)
            reason = self._build_reason(components)

        attentiveness = (1.0 - smoothed_score) * 100.0
        self.attention_history.append(attentiveness)
        attentiveness_smoothed = sum(self.attention_history) / len(self.attention_history)

        should_alarm = self.state == "DROWSY" and smoothed_score >= self.alarm_threshold
        phase = "CALIBRATING" if self.state == "CALIBRATING" else "ACTIVE"
        status = "LEARNING BASELINE" if self.state == "CALIBRATING" else self.state

        return ScoreOutput(
            phase=phase,
            status=status,
            confidence=float(confidence),
            drowsy_score=float(smoothed_score),
            attentiveness=float(attentiveness_smoothed),
            should_alarm=should_alarm,
            reason=reason,
        )
