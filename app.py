import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Vision Pro | Live Detection",
    page_icon="🤖",
    layout="wide"
)

# -------------------- CUSTOM CSS DESIGN --------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: radial-gradient(circle at top left, #1e1b4b, #0f172a);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glassmorphism card for the video stream */
    .video-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
    }

    /* Gradient Text for Title */
    .main-title {
        background: -webkit-linear-gradient(#a78bfa, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 50px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }

    .sub-text {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Customizing info boxes */
    .stAlert {
        border-radius: 12px;
        background-color: rgba(99, 102, 241, 0.2);
        color: #e0e7ff;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR DESIGN --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80) # AI Icon
    st.title("Settings")
    st.markdown("---")
    
    st.subheader("Model Configuration")
    model_option = st.selectbox("Select Model Architecture", ["yolov8n.pt", "yolov8s.pt"], help="Nano is faster, Small is more accurate.")
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35)
    
    st.markdown("---")
    st.subheader("App Info")
    st.info("System uses YOLOv8 for real-time inference. Make sure your room is well-lit.")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_option)

# -------------------- MAIN CONTENT --------------------
st.markdown('<p class="main-title">AI VISION PRO</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Neural Network Object Tracking System</p>', unsafe_allow_html=True)

# Create two columns: one for video, one for stats/instructions
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    
    # Callback function
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inference
        results = model.predict(img, conf=confidence, verbose=False)
        
        # Draw Results
        annotated_frame = results[0].plot()
        
        # Add a custom FPS or Object Count overlay (optional)
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects: {count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Streamer
    webrtc_streamer(
        key="object-detection",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("📈 Live Analytics")
    st.write("Detection is currently active.")
    
    # Status Indicators
    st.success("WebRTC: Connected")
    st.info(f"Active Model: {model_option}")
    
    st.divider()
    
    st.subheader("📖 Quick Instructions")
    st.markdown("""
    1. Click **Start** to begin stream.
    2. Adjust **Confidence** if you see flicker.
    3. Switch to **Small (s)** model for better precision.
    """)

# Footer
st.markdown("---")
st.caption("Developed for AI Hands-on Activity 2026")
