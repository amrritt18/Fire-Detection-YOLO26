import time

import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from src.alert_history import get_alert_history
from src.config import DEFAULT_CONFIDENCE, MODEL_PATH
from src.detector import FireDetector
from src.model import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 2rem;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        text-align: center;
        font-weight: bold;
    }

    .safe {
        background-color: #d4edda;
        color: #155724;
    }

    .danger {
        background-color: #f8d7da;
        color: #721c24;
    }

    .warning {
        background-color: #fff3cd;
        color: #856404;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🔥 Real-Time Fire Detection System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "YOLO26-based real-time fire, smoke and object detection"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()

except Exception as error:
    st.error(f"Failed to load model: {error}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Model:** YOLO26n")
    st.write(f"**Model path:** `{MODEL_PATH}`")

    st.write("**Classes:**")

    st.write("🔥 Fire")
    st.write("💨 Smoke")
    st.write("📦 Other")

    st.divider()

    st.info(
        """
        **Safety Notice**

        This system is a computer-vision prototype.
        It should not be treated as a certified
        fire-safety alarm system.
        """
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = FireDetector(
            model=model,
            confidence=confidence,
        )

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Process frame through detector
        annotated_frame = (
            self.detector.process_frame(img)
        )

        return frame.from_ndarray(
            annotated_frame,
            format="bgr24",
        )


# ============================================================
# DASHBOARD LAYOUT
# ============================================================

camera_col, status_col = st.columns(
    [2.2, 1]
)


# ============================================================
# CAMERA
# ============================================================

with camera_col:

    st.subheader("📷 Live Camera")

    webrtc_ctx = webrtc_streamer(
        key="fire-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )


# ============================================================
# STATUS PANEL
# ============================================================

with status_col:

    st.subheader("📊 Detection Status")

    status_placeholder = st.empty()

    st.divider()

    st.markdown("### Detection Counts")

    fire_metric = st.empty()
    smoke_metric = st.empty()
    other_metric = st.empty()

    st.divider()

    st.markdown("### Detection Details")

    confidence_placeholder = st.empty()
    class_placeholder = st.empty()
    fps_placeholder = st.empty()

    st.divider()

    st.markdown("### Alert Status")

    email_placeholder = st.empty()


# ============================================================
# ALERT HISTORY
# ============================================================

st.divider()

st.subheader("🚨 Alert History")

history_placeholder = st.empty()


# ============================================================
# DASHBOARD UPDATE LOOP
# ============================================================

if webrtc_ctx.video_processor:

    processor = webrtc_ctx.video_processor

    while webrtc_ctx.state.playing:

        try:

            status = processor.detector.get_status()

            fire_count = status["fire_count"]
            smoke_count = status["smoke_count"]
            other_count = status["other_count"]

            max_confidence = status["max_confidence"]
            detected_class = status["detected_class"]
            fps = status["fps"]
            alert_status = status["alert_status"]


            # ------------------------------------------------
            # Overall status
            # ------------------------------------------------

            if fire_count > 0:

                status_placeholder.markdown(
                    """
                    <div class="status-box danger">
                        🔥 FIRE DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif smoke_count > 0:

                status_placeholder.markdown(
                    """
                    <div class="status-box warning">
                        💨 SMOKE DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                status_placeholder.markdown(
                    """
                    <div class="status-box safe">
                        ✅ NO FIRE DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # ------------------------------------------------
            # Detection counts
            # ------------------------------------------------

            fire_metric.metric(
                "🔥 Fire",
                fire_count,
            )

            smoke_metric.metric(
                "💨 Smoke",
                smoke_count,
            )

            other_metric.metric(
                "📦 Other",
                other_count,
            )


            # ------------------------------------------------
            # Detection details
            # ------------------------------------------------

            confidence_placeholder.metric(
                "Highest Confidence",
                f"{max_confidence * 100:.1f}%",
            )

            class_placeholder.metric(
                "Detected Class",
                detected_class.upper(),
            )

            fps_placeholder.metric(
                "FPS",
                f"{fps:.1f}",
            )


            # ------------------------------------------------
            # Email status
            # ------------------------------------------------

            email_placeholder.info(
                alert_status
            )


            # ------------------------------------------------
            # Alert history
            # ------------------------------------------------

            history = get_alert_history(
                limit=10
            )

            if history:

                history_placeholder.dataframe(
                    history,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                history_placeholder.info(
                    "No fire alerts recorded yet."
                )


            time.sleep(0.1)

        except Exception:
            break


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fire Detection System • YOLO26n • Streamlit • WebRTC"
)