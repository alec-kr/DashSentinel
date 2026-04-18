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


class DashSentinelApp:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.profile = DriverProfile(args.profile_path)
        if getattr(args, "rebuild_baseline_on_start", False):
            self.profile.reset_baseline()
        self.profile.begin_session()

        self.extractor = FeatureExtractor(
            ear_threshold=args.ear_threshold,
            yawn_mar_threshold=args.yawn_mar_threshold,
            yawn_frames_threshold=args.yawn_frames_threshold,
        )
        self.scorer = AdaptiveScorer(
            profile=self.profile,
            calibration_seconds=args.calibration_seconds,
            alarm_threshold=args.alarm_threshold,
            attention_window=args.attention_window,
            warning_threshold=args.warning_threshold,
            drowsy_threshold=args.drowsy_threshold,
            status_hold_frames=args.status_hold_frames,
            no_face_hold_frames=args.no_face_hold_frames,
        )
        self.logger = EventLogger(args.log_path, args.log_csv)
        self.alarm = AlarmController(enable_alarm=args.enable_alarm, gpio_pin=args.buzzer_pin)

        self.last_profile_save = time.time()
        self.last_alarm_sent = 0.0
        self.fps_history = deque(maxlen=20)
        self.last_frame_time = time.time()
        self.baseline_frames_collected = 0
        self.baseline_started_at = None

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.running = False

    def _open_camera(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.camera_fps)
        return cap

    def _update_fps(self):
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

    def _draw_overlay(self, frame, score: ScoreOutput, features, fps: float):
        if score.status == "DROWSY":
            color = (0, 0, 255)
        elif score.status == "WARNING":
            color = (0, 165, 255)
        elif score.status == "ALERT":
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)

        cv2.putText(frame, f"phase: {score.phase}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)
        cv2.putText(frame, f"status: {score.status}", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)
        cv2.putText(frame, f"confidence: {score.confidence * 100:.1f}%", (18, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cv2.putText(frame, f"drowsy score: {score.drowsy_score:.3f}", (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2)
        cv2.putText(frame, f"attentiveness: {score.attentiveness:.1f}/100", (18, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2)
        cv2.putText(frame, f"reason: {score.reason}", (18, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if features is not None:
            cv2.putText(frame, f"ear: {features['ear']:.3f}  mar: {features['mar']:.3f}", (18, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)
            cv2.putText(frame, f"blink/min: {features['blink_rate']:.1f}  yawns: {features['yawn_count']}", (18, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)
            cv2.putText(frame, f"roll: {features['roll_deg']:.1f}  yaw: {features['yaw_ratio']:.2f}  pitch: {features['pitch_ratio']:.2f}", (18, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 2)
            cv2.putText(frame, f"closed: {features['closed_frames_norm']:.2f}  yawnflag: {int(features['yawn_flag'])}  posture: {int(features['posture_flag'])}", (18, 304), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 2)

        cv2.putText(frame, f"profile updates: {self.profile.total_updates}", (18, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 2)
        cv2.putText(frame, f"fps: {fps:.1f}", (18, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 2)
        if not self.args.headless:
            cv2.putText(frame, "press q to quit", (18, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 2)

    def _draw_baseline_overlay(self, frame, features, fps, elapsed):
        remaining = max(0.0, self.args.startup_baseline_seconds - elapsed)
        cv2.putText(frame, "phase: startup baseline", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2)
        cv2.putText(frame, "look at the camera with a neutral face", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)
        cv2.putText(frame, f"frames collected: {self.baseline_frames_collected}", (18, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, f"time remaining: {remaining:.1f}s", (18, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        if features is not None:
            cv2.putText(frame, f"ear: {features['ear']:.3f}  mar: {features['mar']:.3f}  blink/min: {features['blink_rate']:.1f}", (18, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)
        cv2.putText(frame, f"fps: {fps:.1f}", (18, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180, 180, 180), 2)
        if self.args.show_ui and not self.args.headless:
            cv2.putText(frame, "press q to quit", (18, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180, 180, 180), 2)

    def _feature_is_valid_for_baseline(self, features):
        return (
            features["closed_frames_norm"] < 0.15
            and features["yawn_flag"] < 0.5
            and features["posture_flag"] < 0.5
        )

    def _startup_baseline_complete(self, elapsed):
        enough_time = elapsed >= self.args.startup_baseline_seconds
        enough_frames = self.baseline_frames_collected >= self.args.startup_baseline_min_frames
        return enough_time and enough_frames

    def _run_startup_baseline(self, cap, face_mesh):
        self.extractor.reset()
        self.baseline_frames_collected = 0
        self.baseline_started_at = time.time()

        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            if self.args.mirror:
                frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (self.args.width, self.args.height), interpolation=cv2.INTER_AREA)
            enhanced = enhance_lighting(frame)
            display_frame = enhanced.copy()
            fps = self._update_fps()

            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            features = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = enhanced.shape[:2]
                features = self.extractor.extract(landmarks, w, h)
                if self._feature_is_valid_for_baseline(features):
                    self.profile.update_from_alert_frame(features)
                    self.baseline_frames_collected += 1

            elapsed = time.time() - self.baseline_started_at
            if self.args.show_ui and not self.args.headless:
                self._draw_baseline_overlay(display_frame, features, fps, elapsed)
                cv2.imshow(self.args.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                    break
            else:
                sys.stdout.write(f"\r[{now_ts()}] building baseline frames={self.baseline_frames_collected} elapsed={elapsed:.1f}s   ")
                sys.stdout.flush()

            if self._startup_baseline_complete(elapsed):
                self.profile.save()
                break

        if not self.args.show_ui or self.args.headless:
            print()
        self.extractor.reset()

    def _trigger_alarm_if_needed(self, score):
        if not score.should_alarm:
            return
        now = time.time()
        if now - self.last_alarm_sent < self.args.alarm_cooldown_seconds:
            return

        duration = self.args.alarm_duration_seconds
        if score.drowsy_score >= 0.85:
            duration *= 2.0
        self.alarm.trigger(duration_s=duration)
        self.last_alarm_sent = now

    def run(self):
        cap = self._open_camera()
        mp_face_mesh = mp.solutions.face_mesh
        frame_count = 0

        try:
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=self.args.refine_landmarks,
                min_detection_confidence=self.args.min_detection_confidence,
                min_tracking_confidence=self.args.min_tracking_confidence,
            ) as face_mesh:
                self._run_startup_baseline(cap, face_mesh)
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
                        phase="ACTIVE",
                        status="NO FACE",
                        confidence=0.0,
                        drowsy_score=0.0,
                        attentiveness=self.scorer.attention_history[-1] if self.scorer.attention_history else 100.0,
                        should_alarm=False,
                        reason="face not visible",
                    )
                    features = None

                    process_this = frame_count % max(1, self.args.process_every_n_frames) == 0
                    if process_this:
                        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb)
                        if results.multi_face_landmarks:
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
                            score = self.scorer.update_no_face()

                    self._trigger_alarm_if_needed(score)

                    if self.args.show_ui and not self.args.headless:
                        self._draw_overlay(display_frame, score, features, fps)
                        cv2.imshow(self.args.window_name, display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            self.running = False
                    else:
                        if process_this:
                            sys.stdout.write(
                                f"\r[{now_ts()}] phase={score.phase} status={score.status} conf={score.confidence:.2f} score={score.drowsy_score:.2f} attentiveness={score.attentiveness:.1f} reason={score.reason}   "
                            )
                            sys.stdout.flush()

                    if time.time() - self.last_profile_save >= self.args.save_profile_every_seconds:
                        self.profile.save()
                        self.last_profile_save = time.time()
        finally:
            self.profile.save()
            self.alarm.cleanup()
            cap.release()
            if self.args.show_ui and not self.args.headless:
                cv2.destroyAllWindows()
            else:
                print()
