import cv2
import numpy as np
import tempfile
from app.backend.models.pose_model import build_model, VideoProcessor

def analyze_session(video_bytes, det_conf=0.5, track_conf=0.5, clf_thr=0.5):
    # Save video to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    model = build_model()
    processor = VideoProcessor(det_conf, track_conf, clf_thr, model)

    cap = cv2.VideoCapture(tmp_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processor.process(frame)
    cap.release()

    return {
        "curl_reps": processor.curl_counter,
        "press_reps": processor.press_counter,
        "squat_reps": processor.squat_counter,
        "detected_action": processor.current_action
    }