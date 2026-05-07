import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

# Cache the model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracking")
st.write("Point your camera at objects to identify them in real-time.")

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ===== VIDEO PROCESSOR =====
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.conf = confidence  # pass slider value

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=img,
            conf=self.conf,
            verbose=False
        )

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ===== STREAM =====
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
