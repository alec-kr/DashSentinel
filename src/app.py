# sleepy guard app
# real-time driver monitoring using face landmarks and adaptive scoring

import signal
import sys
import time
from collections import deque

import cv2
import mediapipe as mp

from .alarm import AlarmController
from .constants import LEFT_EYE, MOUTH_BOTTOM, MOUTH_TOP, NOSE_TIP, RIGHT_EYE
from .features import FeatureExtractor, enhance_lighting
from .logging_utils import EventLogger
from .profile import DriverProfile
from .scoring import AdaptiveScorer, ScoreOutput
from .utils import now_ts


# class that groups related logic/state for this component
class DashSentinelApp:
# initialize class state and configuration
    def __init__(self, args):
        self.args = args
        self.running = True
        self.profile = DriverProfile(args.profile_path)
        self.profile.begin_session()
        self.extractor = FeatureExtractor(
# compute eye aspect ratio to detect eye closure
            ear_threshold=args.ear_threshold,
# compute mouth aspect ratio to detect yawning
            yawn_mar_threshold=args.yawn_mar_threshold,
            yawn_frames_threshold=args.yawn_frames_threshold,
        )
        self.scorer = AdaptiveScorer(
            profile=self.profile,
            calibration_seconds=args.calibration_seconds,
            alarm_threshold=args.alarm_threshold,
            attention_window=args.attention_window,
        )
        self.logger = EventLogger(args.log_path, args.log_csv)
        self.alarm = AlarmController(enable_alarm=args.enable_alarm, gpio_pin=args.buzzer_pin)
# current timestamp used for fps or timing logic
        self.last_profile_save = time.time()
        self.last_alarm_sent = 0.0
        self.fps_history = deque(maxlen=20)
# current timestamp used for fps or timing logic
        self.last_frame_time = time.time()

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

# function that handles a specific step in the pipeline
    def _handle_signal(self, signum, frame):
        self.running = False

# open and configure camera device
    def _open_camera(self):
# create video capture from webcam or camera module
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.camera_fps)
        return cap

# update running statistics / profile with new data
    def _update_fps(self):
# current timestamp used for fps or timing logic
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

# render overlay text/metrics on frame
    def _draw_overlay(self, frame, score: ScoreOutput, features, fps: float):
        color = (0, 0, 255) if score.status == "DROWSY" else (0, 255, 0) if score.status in ("ALERT",) else (0, 255, 255)
        cv2.putText(frame, f"Phase: {score.phase}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)
        cv2.putText(frame, f"Status: {score.status}", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)
        cv2.putText(frame, f"Confidence: {score.confidence * 100:.1f}%", (18, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cv2.putText(frame, f"Drowsy Score: {score.drowsy_score:.3f}", (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2)
        cv2.putText(frame, f"Attentiveness: {score.attentiveness:.1f}/100", (18, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2)

        if features is not None:
            cv2.putText(frame, f"EAR: {features['ear']:.3f}  MAR: {features['mar']:.3f}", (18, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(frame, f"Blink/min: {features['blink_rate']:.1f}  Yawns: {features['yawn_count']}", (18, 224), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll: {features['roll_deg']:.1f}  Yaw: {features['yaw_ratio']:.2f}  Pitch: {features['pitch_ratio']:.2f}", (18, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 2)
            cv2.putText(frame, f"Closed: {features['closed_frames_norm']:.2f}  YawnFlag: {int(features['yawn_flag'])}  Posture: {int(features['posture_flag'])}", (18, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 2)

        cv2.putText(frame, f"Profile updates: {self.profile.total_updates}", (18, 304), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (18, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 2)
        if not self.args.headless:
            cv2.putText(frame, "Press Q to quit", (18, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 2)

# main loop that captures frames and runs detection/scoring
    def run(self):
        cap = self._open_camera()
        mp_face_mesh = mp.solutions.face_mesh
        frame_count = 0

        try:
# mediapipe face mesh model for extracting facial landmarks
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=self.args.refine_landmarks,
                min_detection_confidence=self.args.min_detection_confidence,
                min_tracking_confidence=self.args.min_tracking_confidence,
            ) as face_mesh:
                while self.running:
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.03)
                        continue

                    frame_count += 1
                    if self.args.mirror:
                        frame = cv2.flip(frame, 1)

                    frame = cv2.resize(frame, (self.args.width, self.args.height), interpolation=cv2.INTER_AREA)
                    enhanced = enhance_lighting(frame)
                    display_frame = enhanced.copy()
                    fps = self._update_fps()

                    score = ScoreOutput(
                        phase="NO FACE",
                        status="NO FACE",
                        confidence=0.0,
                        drowsy_score=0.0,
                        attentiveness=self.scorer.attention_history[-1] if self.scorer.attention_history else 100.0,
                        should_alarm=False,
                    )
                    features = None

                    process_this = frame_count % max(1, self.args.process_every_n_frames) == 0
                    if process_this:
# convert image color space (bgr <-> rgb/lab)
                        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb)
                        if results.multi_face_landmarks:
# compute mouth aspect ratio to detect yawning
                            landmarks = results.multi_face_landmarks[0].landmark
                            h, w = enhanced.shape[:2]
                            features = self.extractor.extract(landmarks, w, h)
                            score = self.scorer.score(features)

                            if self.args.draw_landmarks:
                                for idx in LEFT_EYE + RIGHT_EYE + [NOSE_TIP, MOUTH_TOP, MOUTH_BOTTOM]:
                                    lm = landmarks[idx]
                                    x = int(lm.x * w)
                                    y = int(lm.y * h)
                                    cv2.circle(display_frame, (x, y), 2, (255, 255, 255), -1)

                            self.logger.write_periodic(
                                status=score.status,
                                confidence=score.confidence,
                                drowsy_score=score.drowsy_score,
                                attentiveness=score.attentiveness,
                                features=features,
                                every_seconds=1.0,
                            )
                        else:
                            self.extractor.reset()

# current timestamp used for fps or timing logic
                    if score.should_alarm and (time.time() - self.last_alarm_sent) >= self.args.alarm_cooldown_seconds:
                        self.alarm.trigger(duration_s=self.args.alarm_duration_seconds)
# current timestamp used for fps or timing logic
                        self.last_alarm_sent = time.time()

                    if self.args.show_ui and not self.args.headless:
                        self._draw_overlay(display_frame, score, features, fps)
# display frame in a window (for debugging/ui mode)
                        cv2.imshow(self.args.window_name, display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            self.running = False
                    else:
                        if process_this and features is not None:
                            sys.stdout.write(
                                f"\r[{now_ts()}] phase={score.phase} status={score.status} conf={score.confidence:.2f} score={score.drowsy_score:.2f} attentiveness={score.attentiveness:.1f}   "
                            )
                            sys.stdout.flush()

# current timestamp used for fps or timing logic
                    if time.time() - self.last_profile_save >= self.args.save_profile_every_seconds:
                        self.profile.save()
# current timestamp used for fps or timing logic
                        self.last_profile_save = time.time()
        finally:
            self.profile.save()
            self.alarm.cleanup()
            cap.release()
            if self.args.show_ui and not self.args.headless:
                cv2.destroyAllWindows()
            else:
                print()
