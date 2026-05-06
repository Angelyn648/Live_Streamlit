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

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Settings")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

line_position = st.sidebar.slider("Line Position (Y)", 100, 500, 300)

object_type = st.sidebar.selectbox(
    "Object to Count",
    ["All", "Person", "Car"]
)

show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)

# COCO class IDs
CLASS_MAP = {
    "Person": 0,
    "Car": 2
}

# ============== VIDEO PROCESSOR ==============
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.counted_ids = set()
        self.total_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = self.model.track(
            source=img,
            persist=True,
            conf=confidence,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated_frame = img.copy()

        # Draw line
        cv2.line(
            annotated_frame,
            (0, line_position),
            (annotated_frame.shape[1], line_position),
            (0, 255, 0),
            2
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, obj_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)

                # Filter objects
                if object_type != "All":
                    if int(cls) != CLASS_MAP[object_type]:
                        continue

                center_y = int((y1 + y2) / 2)

                # Count crossing
                if center_y > line_position and obj_id not in self.counted_ids:
                    self.counted_ids.add(obj_id)
                    self.total_count += 1

                # Draw bounding box
                if show_boxes:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated_frame,
                        f"ID {int(obj_id)}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        # Display count
        cv2.putText(
            annotated_frame,
            f"Count: {self.total_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ============== STREAM ==============
webrtc_streamer(
    key="tracking",
    video_processor_factory=VideoProcessor,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
