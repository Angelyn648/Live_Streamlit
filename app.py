import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Live Object Detection & Tracing",
    page_icon="🎥",
    layout="wide"
)

# -------------------- STYLE --------------------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9);
        border-right: 1px solid rgba(167, 139, 250, 0.2);
    }

    .title {
        font-size: 38px;
        font-weight: bold;
        text-align: center;
        color: #a78bfa;
        margin-top: -20px;
        text-shadow: 0px 4px 10px rgba(167, 139, 250, 0.3);
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #cbd5f5;
        margin-bottom: 20px;
    }

    video {
        max-width: 50% !important;
        height: auto !important;
        border-radius: 12px;
        margin: 0 auto;
        display: block;
    }

    button[data-testid="stBaseButton-secondary"] {
        padding: 2px 10px !important;
        font-size: 12px !important;
        min-height: 25px !important;
    }

    div[data-testid="stWebRtcDeviceSelect"] {
        aspect-ratio: 1 / 1 !important; 
        max-width: 300px !important;    
        margin: 0 auto;                 
        overflow: hidden;
        border-radius: 15px;
        border: 2px solid rgba(167, 139, 250, 0.5);
    }

    video {
        object-fit: cover !important; 
        width: 50% !important;
        height: 50% !important;
    }

    .stWebRtc {
        gap: 2px !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title"> Live Object Detection & Tracing </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Point your camera at objects to identify them in real-time.</div>', unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("<h3 style='color: #a78bfa;'>⚙️ Settings</h3>", unsafe_allow_html=True)

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)
model_option = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
st.sidebar.write("Lower confidence = more detections.")
st.sidebar.write("Higher confidence = fewer false positives.")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_option)

# -------------------- MAIN --------------------
st.markdown('<div class="video-container">', unsafe_allow_html=True)

st.info("Allow camera access to start.")

# -------------------- CALLBACK --------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=confidence, verbose=False)
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

st.success("Camera ready")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><p style='text-align: center; color: #64748b; font-size: 0.8rem;'>@2026 Live Object Detection System | Developed by: Angelyn V. Sto.Domingo</p>", unsafe_allow_html=True)
