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

    .card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        
        max-width: 450px; 
        margin: 0 auto;
    }

    div[data-testid="stWebRtc"] > div {
        aspect-ratio: 1 / 1 !important; 
        overflow: hidden;
        border-radius: 12px;
    }

    video {
        width: 100% !important;
        height: 100% !important;
        aspect-ratio: 1 / 1 !important;
        object-fit: cover !important; 
        border-radius: 12px;
    }

    button[data-testid="stBaseButton-secondary"] {
        padding: 2px 15px !important;
        font-size: 12px !important;
    }
    
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title"> Live Object Detection & Tracing</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time AI detection using YOLOv8</div>', unsafe_allow_html=True)

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
st.markdown('<div class="card">', unsafe_allow_html=True)

st.info("Allow camera access. Detection will start automatically.")

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

st.success(" Camera ready")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><p style='text-align: center; color: #64748b; font-size: 0.8rem;'>@2026 Live Object Detection System| Develop by: [Angelyn V. Sto.Domingo]</p>", unsafe_allow_html=True)
