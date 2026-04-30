import os
import time

from .utils import ensure_parent, now_ts


# handle logging and features to a CSV file
class EventLogger:
    def __init__(self, path: str, enabled: bool):
        self.path = path
        self.enabled = enabled
        self.last_write = 0.0
        if enabled:
            # create log file with header if it doesn't exist
            ensure_parent(path)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("timestamp,status,confidence,drowsy_score,attentiveness,ear,mar,blink_rate,roll_deg,yaw_ratio,pitch_ratio,closed_frames_norm,yawn_flag,posture_flag,yawn_count\n")

    # periodically update logs with current status and features
    def write_periodic(self, status: str, confidence: float, drowsy_score: float, attentiveness: float, features: dict, every_seconds: float = 1.0):
        if not self.enabled:
            return
        # current timestamp used for fps or timing logic
        now = time.time()
        if now - self.last_write < every_seconds:
            return
        self.last_write = now
        
        # construction of csv row with required data
        row = (
            f"{now_ts()},{status},{confidence:.4f},{drowsy_score:.4f},{attentiveness:.2f},"
            f"{features['ear']:.4f},{features['mar']:.4f},{features['blink_rate']:.2f},"
            f"{features['roll_deg']:.2f},{features['yaw_ratio']:.4f},{features['pitch_ratio']:.4f},"
            f"{features['closed_frames_norm']:.4f},{features['yawn_flag']:.0f},{features['posture_flag']:.0f},{features['yawn_count']}\n"
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(row)
