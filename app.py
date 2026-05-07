import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Live Object Detection",
    page_icon="🎥",
    layout="wide"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #a78bfa;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 20px;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">🎥 Live Object Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time AI detection using YOLOv8</div>', unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Settings")

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)
model_option = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"])

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_option)

# -------------------- MAIN --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.info("📷 Allow camera access. Detection will start automatically.")

# -------------------- CALLBACK (ONLY PROCESS IMAGE) --------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.predict(
        img,
        conf=confidence,
        verbose=False
    )

    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# -------------------- STREAM --------------------
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.success("🟢 Camera ready")

st.markdown('</div>', unsafe_allow_html=True)
