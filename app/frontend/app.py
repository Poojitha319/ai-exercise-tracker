import streamlit as st
import cv2
from tf_keras.models import Model
from tf_keras.layers import LSTM, Dense, Dropout, Input, Flatten, Bidirectional, Permute, multiply
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
import requests

# ---------------- Backend URL ----------------
BACKEND_URL = "http://localhost:8000"  # change when deployed

# ---------------- Project Title ----------------
st.set_page_config(page_title="Exercise Tracker AI", page_icon="💪", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🏋️ Exercise Tracker AI</h1>
    <p style='text-align: center; color: gray;'>An AI-powered trainer to track your exercises, meals, and fitness plans</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
st.sidebar.title("🔌 System Status")
def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/health/", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

if st.sidebar.button("Check Backend Status"):
    if check_backend_status():
        st.sidebar.success("✅ Backend is running")
    else:
        st.sidebar.error("❌ Backend not responding")

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

    load_dir = r"D:\new\new\exercise_tracker\app\models\ml_models\LSTM_Attention.h5"
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Model weights not found at {load_dir}")
    model.load_weights(load_dir)
    return model

HIDDEN_UNITS = 256
model = build_model(HIDDEN_UNITS)

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

    def process(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        self.draw_landmarks(image, results)
        return image

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- WebRTC ----------------
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ---------------- Tabs ----------------
tabs = st.tabs([
    "Exercise Setup",
    "Meal Analyzer",
    "Nutrition Planner",
    "Workout Planner",
    "Exercise Trainer"
])

if "exercise" not in st.session_state:
    st.session_state.exercise = "curl"
if "thresholds" not in st.session_state:
    st.session_state.thresholds = (0.7, 0.7, 0.7)
if "exercises" not in st.session_state:
    st.session_state.exercises = ["curl", "press", "squat"]

# ---------- Tab 1: Exercise Setup ----------
with tabs[0]:
    st.header("⚙️ Exercise Setup")
    choice = st.selectbox(
        "Select Exercise",
        st.session_state.exercises + ["➕ Add via YouTube URL"],
        key="exercise_selector"
    )
    if choice == "➕ Add via YouTube URL":
        st.subheader("Add a New Exercise using YouTube URL")
        youtube_url = st.text_input("Enter YouTube Video URL")
        if st.button("Detect Exercise"):
            if youtube_url.strip():
                try:
                    res = requests.post(f"{BACKEND_URL}/detect_exercise", json={"url": youtube_url}, timeout=10)
                    if res.status_code == 200:
                        detected_ex = res.json().get("exercise_name")
                        if detected_ex:
                            st.success(f"🏷️ Detected Exercise: {detected_ex}")
                            if detected_ex not in st.session_state.exercises:
                                st.session_state.exercises.append(detected_ex)
                                st.success(f"✅ '{detected_ex}' added to list!")
                            else:
                                st.info("ℹ️ Already in exercise list.")
                        else:
                            st.error("❌ Could not detect exercise.")
                    else:
                        st.error(f"❌ Backend error: {res.text}")
                except Exception as e:
                    st.error(f"Backend not reachable: {e}")
            else:
                st.warning("⚠️ Please enter a valid YouTube URL.")
    else:
        st.write(f"✅ Selected Exercise: **{choice}**")
        threshold1 = st.slider("Detection Confidence", 0.0, 1.0, 0.7, 0.05)
        threshold2 = st.slider("Tracking Confidence", 0.0, 1.0, 0.7, 0.05)
        threshold3 = st.slider("Classification Threshold", 0.0, 1.0, 0.7, 0.05)
        st.session_state.thresholds = (threshold1, threshold2, threshold3)
        st.success("✅ Setup saved! Go to 'Exercise Trainer' tab to start.")

# ---------- Tab 2: Meal Analyzer ----------
with tabs[1]:
    st.header("🥗 Meal Analyzer")
    uploaded_file = st.file_uploader("Upload a meal image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Your Meal", use_column_width=True)
        if st.button("Analyze Meal"):
            st.info("🔍 Sending to backend...")
            files = {"file": uploaded_file.getvalue()}
            try:
                res = requests.post(f"{BACKEND_URL}/meals/analyze", files=files)
                if res.status_code == 200:
                    st.success("✅ Analysis complete!")
                    st.json(res.json())
                else:
                    st.error("❌ Error from backend")
            except Exception as e:
                st.error(f"Backend error: {e}")

# ---------- Tab 3: Nutrition Planner ----------
with tabs[2]:
    st.header("🍎 Nutrition Planner")
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    activity_level = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
    if st.button("Generate Nutrition Plan"):
        payload = {"age": age, "weight": weight, "height": height, "activity_level": activity_level}
        try:
            res = requests.post(f"{BACKEND_URL}/nutrition-plans/generate", json=payload)
            if res.status_code == 200:
                st.success("✅ Nutrition Plan Ready")
                st.json(res.json())
            else:
                st.error("❌ Backend error")
        except Exception as e:
            st.error(f"Backend error: {e}")

# ---------- Tab 4: Workout Planner ----------
with tabs[3]:
    st.header("💪 Workout Planner")
    goal = st.selectbox("Your Goal", ["Weight Loss", "Strength Gain", "Muscle Building", "General Fitness"])
    days = st.slider("Workout Days per Week", 1, 7, 4)
    if st.button("Generate Workout Plan"):
        payload = {"goal": goal, "days": days}
        try:
            res = requests.post(f"{BACKEND_URL}/workouts/generate", json=payload)
            if res.status_code == 200:
                st.success("✅ Workout Plan Ready")
                st.json(res.json())
            else:
                st.error("❌ Backend error")
        except Exception as e:
            st.error(f"Backend error: {e}")

# ---------- Tab 5: Exercise Trainer ----------
with tabs[4]:
    st.header("🎥 Exercise Trainer (Live AI)")
    st.info("Click 'Start' and allow webcam access. Do your exercise in front of the camera!")
    threshold1, threshold2, threshold3 = st.session_state.thresholds
    webrtc_streamer(
        key="Exercise Tracker AI",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: VideoProcessor(threshold1, threshold2, threshold3),
        async_processing=True,
    )
    st.markdown("**Note:** Ensure good lighting and a clear background for optimal performance.")
