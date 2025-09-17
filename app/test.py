#this file is real time live streamlit app for exercise detection and rep counting and still have some sort modifications (need to integrate in main app(tab-4 exrcise trainer))

import streamlit as st
import cv2
from tf_keras.models import Model
from tf_keras.layers import LSTM, Dense, Dropout, Input, Flatten, Bidirectional, Permute, multiply
import numpy as np
import mediapipe as mp
import math
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# ---------------- Model ----------------
def attention_block(inputs, time_steps):
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

@st.cache_resource(show_spinner=False)
def build_model(HIDDEN_UNITS=256, sequence_length=30, num_input_values=33*4, num_classes=3):
    inputs = Input(shape=(sequence_length, num_input_values))
    lstm_out = Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True))(inputs)
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    x = Dense(2*HIDDEN_UNITS, activation='relu')(attention_mul)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=x)

    load_dir = ".app\backend\models\ml_models\LSTM_Attention.h5"
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Model weights not found at {load_dir}")
    model.load_weights(load_dir)
    return model

HIDDEN_UNITS = 256
model = build_model(HIDDEN_UNITS)

# ---------------- UI ----------------
st.title("AI Personal Fitness Trainer Web App")
st.markdown(
    """
    This app uses your webcam and AI to detect your exercise (curl, press, squat) and count your reps in real time!
    """
)

st.sidebar.header("Settings")
threshold1 = st.sidebar.slider("Minimum Keypoint Detection Confidence", 0.00, 1.00, 0.50)
threshold2 = st.sidebar.slider("Minimum Tracking Confidence", 0.00, 1.00, 0.50)
threshold3 = st.sidebar.slider("Minimum Activity Classification Confidence", 0.00, 1.00, 0.50)

st.write("## Activate the AI 🤖🏋️‍♂️")
st.info("Click 'Start' below and allow webcam access. Do your exercise in front of the camera!")

# ---------------- Processor ----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor:
    def __init__(self, det_conf: float, track_conf: float, clf_thr: float):
        self.actions = np.array(['curl', 'press', 'squat'])
        self.sequence_length = 30
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        self.threshold = clf_thr

        self.sequence = []
        self.current_action = ''

        self.curl_counter = 0
        self.press_counter = 0
        self.squat_counter = 0
        self.curl_stage = None
        self.press_stage = None
        self.squat_stage = None

        # create mediapipe pose once per processor
        self.pose = mp_pose.Pose(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )

    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
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
        coord = getattr(mp_pose.PoseLandmark, f"{side.upper()}_{joint.upper()}")
        return [landmarks[coord.value].x, landmarks[coord.value].y]

    def viz_joint_angle(self, image, angle, joint_xy):
        cv2.putText(
            image, str(int(angle)),
            tuple(np.multiply(joint_xy, [image.shape[1], image.shape[0]]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

    def count_reps(self, image, landmarks):
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
            self.viz_joint_angle(image, angle, elbow)

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
            self.viz_joint_angle(image, elbow_angle, elbow)

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

            self.viz_joint_angle(image, left_knee_angle, left_knee)
            self.viz_joint_angle(image, left_hip_angle, left_hip)

    def prob_viz(self, res, input_frame):
        output = input_frame.copy()
        h = 30
        for i, prob in enumerate(res):
            y1, y2 = 60 + i*40, 90 + i*40
            x2 = int(min(max(prob, 0.0), 1.0) * input_frame.shape[1])
            cv2.rectangle(output, (0, y1), (x2, y2), self.colors[i], -1)
            cv2.putText(output, self.actions[i], (10, y2-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return output

    def process(self, image):
        image.flags.writeable = False
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        image.flags.writeable = True
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.draw_landmarks(image, results)

        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints.astype('float32', casting='same_kind'))
        self.sequence = self.sequence[-self.sequence_length:]

        if len(self.sequence) == self.sequence_length:
            res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            self.current_action = self.actions[int(np.argmax(res))]
            conf = float(np.max(res))
            if conf < self.threshold:
                self.current_action = ''
            image = self.prob_viz(res, image)

            if results.pose_landmarks:
                try:
                    self.count_reps(image, results.pose_landmarks.landmark)
                except Exception:
                    pass

            # Top bar with counters
            bar_color = self.colors[int(np.argmax(res))]
            cv2.rectangle(image, (0, 0), (image.shape[1], 40), bar_color, -1)
            cv2.putText(image, f'curl {self.curl_counter}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f'press {self.press_counter}', (240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f'squat {self.squat_counter}', (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        return image

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- WebRTC ----------------
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if st.button("Start Live Session"):
    webrtc_streamer(
        key="AI-trainer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: VideoProcessor(threshold1, threshold2, threshold3),
        async_processing=True,
    )