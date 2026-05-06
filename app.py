import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Object Detection + Counting + Line Crossing")

# Line position (pwede mo baguhin)
LINE_Y = 300

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.counted_ids = set()
        self.total_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run tracking
        results = self.model.track(
            source=img,
            persist=True,
            conf=0.5,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Draw counting line
        cv2.line(annotated_frame, (0, LINE_Y), (annotated_frame.shape[1], LINE_Y), (0, 255, 0), 2)

        # Get boxes and IDs
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)

                center_y = int((y1 + y2) / 2)

                # Check if object crosses the line
                if center_y > LINE_Y and obj_id not in self.counted_ids:
                    self.counted_ids.add(obj_id)
                    self.total_count += 1

        # Display count
        cv2.putText(
            annotated_frame,
            f"Count: {self.total_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


webrtc_streamer(
    key="tracking",
    video_processor_factory=VideoProcessor,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
