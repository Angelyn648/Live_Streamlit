import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import numpy as np

# Page Configuration
st.set_page_config(page_title="AI Object Tracker", layout="wide")

@st.cache_resource
def load_model():
    # Using yolov8n (nano) for best real-time CPU performance
    return YOLO("yolov8n.pt")

model = load_model()

# --- SIDEBAR UI ---
st.sidebar.title("🛠️ Control Panel")
st.sidebar.markdown("Use these settings to adjust the AI detection in real-time.")

# Confidence slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, help="Lower values detect more objects but may have more false positives.")

# Object selection (Enhancement)
all_classes = list(model.names.values())
selected_objects = st.sidebar.multiselect(
    "Filter Objects", 
    options=all_classes, 
    default=["person", "cell phone", "bottle", "laptop"],
    help="Select which objects the AI should look for."
)

st.sidebar.divider()
st.sidebar.subheader("Activity Status")
# Placeholder for status (will be updated by the logic below)
status_placeholder = st.sidebar.empty()

# --- MAIN UI ---
st.title("🎥 Enhanced Live Object Detection & Tracing")
st.write("This application identifies and tracks everyday objects using the YOLOv8 model.")

# Video frame callback logic
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Get class IDs for selected objects
    selected_ids = [k for k, v in model.names.items() if v in selected_objects]

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=conf_threshold,
        classes=selected_ids if selected_ids else None,
        verbose=False
    )

    # Annotate frame
    annotated_frame = results[0].plot()
    
    # ENHANCEMENT: Object Counting Logic
    if results[0].boxes is not None:
        count = len(results[0].boxes)
    else:
        count = 0

    # Overlay count on the video feed
    cv2.putText(
        annotated_frame, 
        f"Objects Detected: {count}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (0, 255, 0), 
        2
    )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Start WebRTC streamer
ctx = webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

# Update the sidebar status based on camera state
if ctx.state.playing:
    status_placeholder.success("✅ Camera Active")
else:
    status_placeholder.warning("⚠️ Camera Offline")
