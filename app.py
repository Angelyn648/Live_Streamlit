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
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
    }

    /* Bawasan ang padding sa taas para sumama ang title sa screenshot */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9);
        border-right: 1px solid rgba(167, 139, 250, 0.2);
    }

    /* Title Styling - Pinaliit nang kaunti para compact */
    .title {
        font-size: 35px;
        font-weight: bold;
        text-align: center;
        color: #a78bfa;
        margin-top: -10px;
        text-shadow: 0px 4px 10px rgba(167, 139, 250, 0.3);
    }

    .subtitle {
        text-align: center;
        font-size: 14px;
        color: #cbd5f5;
        margin-bottom: 10px;
    }

    /* SQUARE VIDEO CONTAINER */
    /* Target ang wrapper ng WebRTC para maging square */
    div[data-testid="stWebRtc"] > div {
        aspect-ratio: 1 / 1 !important;
        max-width: 320px !important; /* Liit na sakto sa screenshot */
        margin: 0 auto;
        overflow: hidden;
        border-radius: 15px;
        border: 2px solid rgba(167, 139, 250, 0.3);
    }

    video {
        aspect-ratio: 1 / 1 !important;
        object-fit: cover !important; /* Ito ang nag-o-crop para maging square */
        width: 100% !important;
        height: 100% !important;
    }

    /* Pinaliit ang control buttons sa ilalim */
    .stWebRtc {
        max-width: 320px !important;
        margin: 0 auto;
    }
    
    button[data-testid="stBaseButton-secondary"] {
        font-size: 11px !important;
        min-height: 25px !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">🎥 Live Object Detection & Tracing</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Point your camera at objects to identify them in real-time.</div>', unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("<h3 style='color: #a78bfa;'>⚙️ Settings</h3>", unsafe_allow_html=True)
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)
model_option = st.sidebar.selectbox("Model", ["yolov8n.pt", "yolov8s.pt"])

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_option)

# -------------------- CALLBACK --------------------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=confidence, verbose=False)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# -------------------- MAIN STREAM --------------------
# Wala nang extra <div> para hindi magka-gap ang layout
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

# Footer
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.7rem; margin-top: 10px;'>@2026 Developed by: Angelyn V. Sto.Domingo</p>", unsafe_allow_html=True)
