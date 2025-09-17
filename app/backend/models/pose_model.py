import os
import numpy as np
from tf_keras.models import Model
from tf_keras.layers import LSTM, Dense, Dropout, Input, Flatten, Bidirectional, Permute, multiply
import mediapipe as mp
import math
import cv2

def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def build_model(HIDDEN_UNITS=256, sequence_length=30, num_input_values=33*4, num_classes=3):
    inputs = Input(shape=(sequence_length, num_input_values))
    lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=x)

    load_dir = "./models/ml_models/LSTM_Attention.h5"
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Model weights not found at {load_dir}")
    model.load_weights(load_dir)
    return model

class VideoProcessor:
    def __init__(self, det_conf: float, track_conf: float, clf_thr: float, model):
        self.actions = np.array(['curl', 'press', 'squat'])
        self.sequence_length = 30
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.threshold = clf_thr
        self.model = model

        self.sequence = []
        self.current_action = ''

        self.curl_counter = 0
        self.press_counter = 0
        self.squat_counter = 0
        self.curl_stage = None
        self.press_stage = None
        self.squat_stage = None

        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )

    def extract_keypoints(self, results):
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility]
                             for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33*4, dtype=np.float32)
        return pose

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def get_coordinates(self, landmarks, side, joint):
        coord = getattr(mp.solutions.pose.PoseLandmark, f"{side.upper()}_{joint.upper()}")
        return [landmarks[coord.value].x, landmarks[coord.value].y]

    def count_reps(self, landmarks):
        if self.current_action == 'curl':
            shoulder = self.get_coordinates(landmarks, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, 'left', 'wrist')
            angle = self.calculate_angle(shoulder, elbow, wrist)
            if angle < 30:
                self.curl_stage = "up"
            if angle > 140 and self.curl_stage == 'up':
                self.curl_stage = "down"
                self.curl_counter += 1
            self.press_stage = None
            self.squat_stage = None

        elif self.current_action == 'press':
            shoulder = self.get_coordinates(landmarks, 'left', 'shoulder')
            elbow = self.get_coordinates(landmarks, 'left', 'elbow')
            wrist = self.get_coordinates(landmarks, 'left', 'wrist')
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            shoulder2elbow_dist = abs(math.dist(shoulder, elbow))
            shoulder2wrist_dist = abs(math.dist(shoulder, wrist))
            if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
                self.press_stage = "up"
            if (elbow_angle < 50) and (shoulder2elbow_dist > shoulder2wrist_dist) and (self.press_stage == 'up'):
                self.press_stage = 'down'
                self.press_counter += 1
            self.curl_stage = None
            self.squat_stage = None

        elif self.current_action == 'squat':
            left_shoulder = self.get_coordinates(landmarks, 'left', 'shoulder')
            left_hip = self.get_coordinates(landmarks, 'left', 'hip')
            left_knee = self.get_coordinates(landmarks, 'left', 'knee')
            left_ankle = self.get_coordinates(landmarks, 'left', 'ankle')
            right_shoulder = self.get_coordinates(landmarks, 'right', 'shoulder')
            right_hip = self.get_coordinates(landmarks, 'right', 'hip')
            right_knee = self.get_coordinates(landmarks, 'right', 'knee')
            right_ankle = self.get_coordinates(landmarks, 'right', 'ankle')

            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)

            thr = 165
            if (left_knee_angle < thr and right_knee_angle < thr and
                left_hip_angle < thr and right_hip_angle < thr):
                self.squat_stage = "down"
            if (left_knee_angle > thr and right_knee_angle > thr and
                left_hip_angle > thr and right_hip_angle > thr and
                self.squat_stage == 'down'):
                self.squat_stage = 'up'
                self.squat_counter += 1
            self.curl_stage = None
            self.press_stage = None

    def process(self, image):
        image.flags.writeable = False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        image.flags.writeable = True

        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints.astype('float32', casting='same_kind'))
        self.sequence = self.sequence[-self.sequence_length:]

        if len(self.sequence) == self.sequence_length:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            self.current_action = self.actions[int(np.argmax(res))]
            conf = float(np.max(res))
            if conf < self.threshold:
                self.current_action = ''
            if results.pose_landmarks:
                try:
                    self.count_reps(results.pose_landmarks.landmark)
                except Exception:
                    pass