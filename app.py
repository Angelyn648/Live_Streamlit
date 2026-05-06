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

# Create a video processor class (IMPORTANT FIX)
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 detection (use predict instead of track first)
        results = self.model.predict(
            source=img,
            conf=0.5,
            verbose=False
        )

        # Draw results
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Start WebRTC
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,  # FIXED
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
