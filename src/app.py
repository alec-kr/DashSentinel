"""
Main application logic.
Initializes the camera feed, processes video frames to extract facial features,
computes drowsiness scores based on an adaptive profile,
and manages the overall state of the application including logging and telemetry.
"""

import signal
import sys
import time
from collections import deque

import cv2
import mediapipe as mp

from .constants import LEFT_EYE, MOUTH_BOTTOM, MOUTH_TOP, NOSE_TIP, RIGHT_EYE
from .features import FeatureExtractor, enhance_lighting
from .logging_utils import EventLogger
from .profile import DriverProfile
from .scoring import AdaptiveScorer, ScoreOutput
from .serial_telemetry import SerialTelemetry
from .utils import now_ts


class DashSentinelApp:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.profile = DriverProfile(args.profile_path)

        # clear any existing baseline if requested, otherwise continue building on it
        if getattr(args, "rebuild_baseline_on_start", False):
            self.profile.reset_baseline()

        # start a new session to track updates in this run
        self.profile.begin_session()

        # setup feature thresholds
        self.extractor = FeatureExtractor(
            ear_threshold=args.ear_threshold,
            yawn_mar_threshold=args.yawn_mar_threshold,
            yawn_frames_threshold=args.yawn_frames_threshold,
        )

        # setup scoring system with adaptive thresholds
        self.scorer = AdaptiveScorer(
            profile=self.profile,
            calibration_seconds=args.calibration_seconds,
            attention_window=args.attention_window,
            warning_threshold=args.warning_threshold,
            drowsy_threshold=args.drowsy_threshold,
            status_hold_frames=args.status_hold_frames,
            no_face_hold_frames=args.no_face_hold_frames,
        )

        # setup the logger to record events and features
        self.logger = EventLogger(args.log_path, args.log_csv)

        # setup telemetry to send data to esp8266 and receive commands
        self.telemetry = SerialTelemetry(
            enabled=getattr(args, "enable_esp_serial", False),
            port=getattr(
                args, "esp_port", "/dev/ttyUSB0"
            ),  # can change with flag --esp-port
            baud=getattr(args, "esp_baud", 115200),
            interval=getattr(args, "esp_send_interval", 0.5),
        )

        # track when we last saved the profile to avoid saving too frequently
        self.last_profile_save = time.time()

        self.fps_history = deque(maxlen=60)
        self.last_frame_time = time.time()
        self.baseline_frames_collected = 0
        self.baseline_started_at = None

        # setup signal handlers for graceful shutdown in terminal
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.running = False

    # helper method to open the camera with the specified settings
    # camera can be changed with the --camera flag (default 0 for primary webcam)
    def _open_camera(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.camera_fps)
        return cap

    # helper method to update FPS calculation based on frame processing time
    def _update_fps(self):
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        # return average FPS over history
        return (
            sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0
        )

    # helper method to draw the overlay with current status and features
    def _draw_overlay(self, frame, score: ScoreOutput, features, fps: float):
        # color coding for different statuses
        if score.status == "DROWSY":
            color = (0, 0, 255)
        elif score.status == "WARNING":
            color = (0, 165, 255)
        elif score.status == "ALERT":
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)

        # draw the main status box
        cv2.putText(
            frame,
            f"phase: {score.phase}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"status: {score.status}",
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"confidence: {score.confidence * 100:.1f}%",
            (18, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"drowsy score: {score.drowsy_score:.3f}",
            (18, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"attentiveness: {score.attentiveness:.1f}/100",
            (18, 162),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"reason: {score.reason}",
            (18, 194),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # if face is detected, show key features and metrics
        if features is not None:
            cv2.putText(
                frame,
                f"ear: {features['ear']:.3f}  mar: {features['mar']:.3f}",
                (18, 226),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"blink/min: {features['blink_rate']:.1f}  yawns: {features['yawn_count']}",
                (18, 252),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                (
                    f"roll: {features['roll_deg']:.1f}"
                    f"yaw: {features['yaw_ratio']:.2f}"
                    f"pitch: {features['pitch_ratio']:.2f}"
                ),
                (18, 278),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 220, 220),
                2,
            )
            cv2.putText(
                frame,
                (
                    f"closed: {features['closed_frames_norm']:.2f}"
                    f"pose: {features.get('bad_pose_norm', 0.0):.2f}"
                    f"away: {features.get('look_away_norm', 0.0):.2f}"
                ),
                (18, 304),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 220, 220),
                2,
            )
        # show profile update count and FPS for debugging and performance monitoring
        cv2.putText(
            frame,
            f"profile updates: {self.profile.total_updates}",
            (18, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            frame,
            f"fps: {fps:.1f}",
            (18, 354),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            2,
        )
        if not self.args.headless:
            cv2.putText(
                frame,
                "press q to quit",
                (18, 378),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (180, 180, 180),
                2,
            )

    # overlay for the startup baseline phase with instructions and progress
    def _draw_baseline_overlay(self, frame, features, fps, elapsed):
        # the remaining time is just for user feedback, it doesn't affect when
        # the baseline phase actually ends since we also require a minimum number of valid frames to be collected
        remaining = max(0.0, self.args.startup_baseline_seconds - elapsed)

        # during baseline collection, we want to encourage the user to look at the camera with a neutral face
        # and avoid excessive blinking or yawning, since those frames won't be as useful for building an accurate profile
        cv2.putText(
            frame,
            "phase: startup baseline",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "look at the camera with a neutral face",
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"frames collected: {self.baseline_frames_collected}",
            (18, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"time remaining: {remaining:.1f}s",
            (18, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

        # show key features to help the user understand what the system is seeing and encourage them to maintain a good baseline posture
        if features is not None:
            cv2.putText(
                frame,
                f"ear: {features['ear']:.3f}  mar: {features['mar']:.3f}  blink/min: {features['blink_rate']:.1f}",
                (18, 158),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 255, 255),
                2,
            )
        cv2.putText(
            frame,
            f"fps: {fps:.1f}",
            (18, 188),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (180, 180, 180),
            2,
        )

        if self.args.show_ui and not self.args.headless:
            cv2.putText(
                frame,
                "press q to quit",
                (18, 214),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (180, 180, 180),
                2,
            )

    # during baseline phase, only use frames where the driver is looking at the camera with a neutral expression and not excessively blinking or yawning
    def _feature_is_valid_for_baseline(self, features):
        return (
            features["closed_frames_norm"] < 0.15
            and features["yawn_flag"] < 0.5
            and features["posture_flag"] < 0.5
        )

    # the baseline phase should only end once we've collected enough valid frames and enough time has passed to ensure a good representation of the driver's normal state
    def _startup_baseline_complete(self, elapsed):
        enough_time = elapsed >= self.args.startup_baseline_seconds
        enough_frames = (
            self.baseline_frames_collected >= self.args.startup_baseline_min_frames
        )
        return enough_time and enough_frames

    # run initial baseline collection to build the profile, then process the frames to update scores
    def _run_startup_baseline(self, cap, face_mesh):
        self.extractor.reset()  # reset to clear any existing states
        self.baseline_frames_collected = 0
        self.baseline_started_at = time.time()

        while self.running:
            ok, frame = cap.read()

            # safety check to avoid processing if camera read fails
            if not ok:
                time.sleep(0.03)
                continue

            if self.args.mirror:
                # flip frame horizontally
                frame = cv2.flip(frame, 1)

            # downscaling for faster baseline processing
            frame = cv2.resize(
                frame, (self.args.width, self.args.height), interpolation=cv2.INTER_AREA
            )
            enhanced = enhance_lighting(frame)
            display_frame = enhanced.copy()
            fps = self._update_fps()

            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            # process frames with mediapipe
            results = face_mesh.process(rgb)
            # reset features unless we get face features in this frame
            features = None

            # only use frames with detected face and valid features
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = enhanced.shape[:2]
                features = self.extractor.extract(landmarks, w, h)
                if self._feature_is_valid_for_baseline(features):
                    self.profile.update_from_alert_frame(features)
                    self.baseline_frames_collected += 1

            # show overlay with progress during baseline phase, or print to console if UI is disabled
            elapsed = time.time() - self.baseline_started_at
            if self.args.show_ui and not self.args.headless:
                self._draw_baseline_overlay(display_frame, features, fps, elapsed)
                cv2.imshow(self.args.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                    break
            else:
                sys.stdout.write(
                    f"\r[{now_ts()}] building baseline frames={self.baseline_frames_collected} elapsed={elapsed:.1f}s   "
                )
                sys.stdout.flush()

            if self._startup_baseline_complete(elapsed):
                self.profile.save()
                break

        if not self.args.show_ui or self.args.headless:
            print()

        # reset extractor to start fresh for active phase
        self.extractor.reset()

    def run(self):
        cap = self._open_camera()
        mp_face_mesh = mp.solutions.face_mesh
        frame_count = 0

        try:
            # create the facemesh object
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

                    # downscale frame for faster processing...mediapipe can be a bottleneck at higher resolutions
                    frame = cv2.resize(
                        frame,
                        (self.args.width, self.args.height),
                        interpolation=cv2.INTER_AREA,
                    )
                    # lighting enhancement for varying light conditions
                    enhanced = enhance_lighting(frame)

                    display_frame = enhanced.copy()
                    fps = self._update_fps()

                    # trigger a warning if no face is detected
                    score = ScoreOutput(
                        phase="ACTIVE",
                        status="NO FACE",
                        confidence=0.0,
                        drowsy_score=0.0,
                        attentiveness=self.scorer.attention_history[-1]
                        if self.scorer.attention_history
                        else 100.0,
                        reason="face not visible",
                    )
                    features = None

                    # process every nth frame. use arg --process-every-n-frames
                    process_this = (
                        frame_count % max(1, self.args.process_every_n_frames) == 0
                    )

                    if process_this:
                        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb)
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0].landmark
                            h, w = enhanced.shape[:2]
                            features = self.extractor.extract(landmarks, w, h)
                            score = self.scorer.score(features)

                            # draw landmarks with flag --draw-landmarks for debugging/visualization
                            if self.args.draw_landmarks:
                                for idx in (
                                    LEFT_EYE
                                    + RIGHT_EYE
                                    + [NOSE_TIP, MOUTH_TOP, MOUTH_BOTTOM]
                                ):
                                    lm = landmarks[idx]
                                    x = int(lm.x * w)
                                    y = int(lm.y * h)
                                    cv2.circle(
                                        display_frame, (x, y), 2, (255, 255, 255), -1
                                    )

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

                    # send score and attentiveness to esp8266, and check for any incoming commands
                    self.telemetry.send(
                        score.status, score.attentiveness, score.drowsy_score
                    )
                    cmd = self.telemetry.read_esp_command()

                    # resetting baseline clears all existing profile data and starts over
                    if cmd == "RESET_BASELINE":
                        self.profile.reset_baseline()
                        self.profile.save()
                        self.scorer.start_time = time.time()
                        print("baseline reset requested from esp8266")

                    # resetting stats clears the current state of all features and scores, but keeps the existing baseline profile data
                    elif cmd == "RESET_STATS":
                        self.scorer.reset_stats()
                        self.extractor.reset_stats()

                        # reset all features that have been tracked over time
                        score = ScoreOutput(
                            phase="ACTIVE",
                            status="ALERT",
                            confidence=1.0,
                            drowsy_score=0.0,
                            attentiveness=100.0,
                            reason="stats reset",
                        )

                        print("full stats reset requested from esp8266")

                    # enabled with --show-ui flag, this draws the overlay on the video feed
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

                    # save profile periodically
                    if (
                        time.time() - self.last_profile_save
                        >= self.args.save_profile_every_seconds
                    ):
                        self.profile.save()
                        self.last_profile_save = time.time()
        finally:
            # ensure we save the profile and close resources on exit
            self.profile.save()
            self.telemetry.close()
            cap.release()
            if self.args.show_ui and not self.args.headless:
                cv2.destroyAllWindows()
            else:
                print()
