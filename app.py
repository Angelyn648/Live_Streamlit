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

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9);
        border-right: 1px solid rgba(167, 139, 250, 0.2);
    }

    /* Original Title Styling */
    .title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        color: #a78bfa;
        margin-top: -20px;
        text-shadow: 0px 4px 10px rgba(167, 139, 250, 0.3);
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5f5;
        margin-bottom: 30px;
    }

    /* Glassmorphism Card for Video */
    .card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
    }

    /* Video Frame Glow */
    iframe {
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(167, 139, 250, 0.1);
    }
    
    .stAlert {
        border-radius: 10px;
    }
    
    video {
        max-width: 0px !important; 
        height: auto !important;
        margin: 0 auto;
        display: block;
        border-radius: 10px;
    }

    .stWebRtc {
        width: 0px !important;
        margin: 0 auto !important;
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
