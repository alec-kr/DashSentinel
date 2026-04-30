# sleepy guard app
# real-time driver monitoring using face landmarks and adaptive scoring

import time
from collections import deque

import cv2
import numpy as np

from .constants import (
    LEFT_CHEEK,
    LEFT_EYE,
    LEFT_EYE_OUTER,
    LEFT_MOUTH,
    MOUTH_BOTTOM,
    MOUTH_TOP,
    NOSE_TIP,
    RIGHT_CHEEK,
    RIGHT_EYE,
    RIGHT_EYE_OUTER,
    RIGHT_MOUTH,
)
from .utils import clamp, euclidean


# function that handles a specific step in the pipeline
def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


# function that handles a specific step in the pipeline
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# function that handles a specific step in the pipeline
def mouth_aspect_ratio(landmarks, w, h):
    top = get_point(landmarks, MOUTH_TOP, w, h)
    bottom = get_point(landmarks, MOUTH_BOTTOM, w, h)
    left = get_point(landmarks, LEFT_MOUTH, w, h)
    right = get_point(landmarks, RIGHT_MOUTH, w, h)

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


# function that handles a specific step in the pipeline
def estimate_head_pose(landmarks, w, h):
    left_eye = get_point(landmarks, LEFT_EYE_OUTER, w, h)
    right_eye = get_point(landmarks, RIGHT_EYE_OUTER, w, h)
    nose = get_point(landmarks, NOSE_TIP, w, h)
    mouth_top = get_point(landmarks, MOUTH_TOP, w, h)
    mouth_bottom = get_point(landmarks, MOUTH_BOTTOM, w, h)
    left_cheek = get_point(landmarks, LEFT_CHEEK, w, h)
    right_cheek = get_point(landmarks, RIGHT_CHEEK, w, h)

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (mouth_top + mouth_bottom) / 2.0

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll_deg = np.degrees(np.arctan2(dy, dx)) if abs(dx) > 1e-6 else 0.0

    left_dist = euclidean(nose, left_cheek)
    right_dist = euclidean(nose, right_cheek)
    yaw_ratio = 0.0
    if (left_dist + right_dist) > 1e-6:
        yaw_ratio = (right_dist - left_dist) / (right_dist + left_dist)

    face_vertical = max(mouth_center[1] - eye_center[1], 1e-6)
    pitch_ratio = (nose[1] - eye_center[1]) / face_vertical
    return float(roll_deg), float(yaw_ratio), float(pitch_ratio)


# function that handles a specific step in the pipeline
def enhance_lighting(frame):
# convert image color space (bgr <-> rgb/lab)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
# convert image color space (bgr <-> rgb/lab)
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    gamma = 1.12
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in np.arange(256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    enhanced = cv2.bilateralFilter(enhanced, 5, 35, 35)
    return enhanced


# class that groups related logic/state for this component
class FeatureExtractor:
# initialize class state and configuration
    def __init__(self, ear_threshold=0.23, yawn_mar_threshold=0.45, yawn_frames_threshold=12):
# compute eye aspect ratio to detect eye closure
        self.ear_threshold = ear_threshold
        self.yawn_mar_threshold = yawn_mar_threshold
        self.yawn_frames_threshold = yawn_frames_threshold
# compute eye aspect ratio to detect eye closure
        self.ear_history = deque(maxlen=10)
        self.blink_timestamps = deque(maxlen=90)
        self.eye_closed = False
        self.closed_frames = 0
        self.mouth_open_frames = 0
        self.last_yawn_time = 0.0
        self.yawn_count = 0

# function that handles a specific step in the pipeline
    def reset(self):
        self.ear_history.clear()
        self.blink_timestamps.clear()
        self.eye_closed = False
        self.closed_frames = 0
        self.mouth_open_frames = 0

# compute features from face landmarks (ear, mar, blink, pose)
    def extract(self, landmarks, w, h):
# compute eye aspect ratio to detect eye closure
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
# compute eye aspect ratio to detect eye closure
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
# compute eye aspect ratio to detect eye closure
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)
# compute eye aspect ratio to detect eye closure
        ear_smoothed = sum(self.ear_history) / len(self.ear_history)

# current timestamp used for fps or timing logic
        now = time.time()
        if ear_smoothed < self.ear_threshold:
            if not self.eye_closed:
                self.eye_closed = True
            self.closed_frames += 1
        else:
            if self.eye_closed:
                self.blink_timestamps.append(now)
                self.eye_closed = False
            self.closed_frames = max(0, self.closed_frames - 2)

        while self.blink_timestamps and now - self.blink_timestamps[0] > 60:
            self.blink_timestamps.popleft()
        blink_rate = float(len(self.blink_timestamps))

# compute mouth aspect ratio to detect yawning
        mar = mouth_aspect_ratio(landmarks, w, h)
        if mar > self.yawn_mar_threshold:
            self.mouth_open_frames += 1
        else:
            if self.mouth_open_frames >= self.yawn_frames_threshold and now - self.last_yawn_time > 2.0:
                self.yawn_count += 1
                self.last_yawn_time = now
            self.mouth_open_frames = 0

        yawn_flag = 1.0 if self.mouth_open_frames >= self.yawn_frames_threshold else 0.0
        roll_deg, yaw_ratio, pitch_ratio = estimate_head_pose(landmarks, w, h)
        posture_flag = 1.0 if abs(roll_deg) > 14 or abs(yaw_ratio) > 0.12 or pitch_ratio < 0.47 or pitch_ratio > 0.72 else 0.0
        closed_frames_norm = clamp(self.closed_frames / 30.0, 0.0, 1.0)

        return {
            "ear": float(ear_smoothed),
            "mar": float(mar),
            "blink_rate": float(blink_rate),
            "roll_deg": float(roll_deg),
            "yaw_ratio": float(yaw_ratio),
            "pitch_ratio": float(pitch_ratio),
            "closed_frames_norm": float(closed_frames_norm),
            "yawn_flag": float(yawn_flag),
            "posture_flag": float(posture_flag),
            "yawn_count": int(self.yawn_count),
        }
    
    def reset_stats(self):
        self.reset()
        self.yawn_count = 0
        self.last_yawn_time = 0.0
