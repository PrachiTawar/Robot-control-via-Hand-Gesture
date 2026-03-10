"""
app.py
------
Streamlit front-end for the AI Gesture Controlled Robot Simulator.

Run with:
    streamlit run app.py
"""

import time
import cv2
import numpy as np
import streamlit as st

from gesture_control import (
    GestureDetector,
    GestureResult,
    draw_command_overlay,
    COMMAND_NONE,
    COMMAND_FORWARD,
    COMMAND_BACKWARD,
    COMMAND_LEFT,
    COMMAND_RIGHT,
    COMMAND_STOP,
)

# ─────────────────────────────────────────────
#  Page configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Gesture Controlled Robot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  Custom CSS – clean, dark-accent theme
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00c896, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        color: #8899aa;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    /* Command card */
    .command-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border-radius: 16px;
        padding: 24px 32px;
        text-align: center;
        border: 1px solid #2a3a5a;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        margin-bottom: 16px;
    }
    .command-label {
        font-size: 0.85rem;
        color: #667788;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .command-text {
        font-size: 2rem;
        font-weight: 700;
        color: #00e5b0;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .status-active   { background:#0d3d2d; color:#00e5a0; border:1px solid #00c880; }
    .status-inactive { background:#2a1a1a; color:#cc4444; border:1px solid #993333; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12161f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "camera_running": False,
        "current_command": COMMAND_NONE,
        "frame_count": 0,
        "start_time": None,
        "fps": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ─────────────────────────────────────────────
#  Sidebar – controls & gesture reference
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    # Start / Stop button
    btn_label = "⏹ Stop Camera" if st.session_state.camera_running else "▶ Start Camera"
    btn_type  = "primary" if not st.session_state.camera_running else "secondary"

    if st.button(btn_label, type=btn_type, use_container_width=True):
        st.session_state.camera_running = not st.session_state.camera_running
        if st.session_state.camera_running:
            st.session_state.start_time  = time.time()
            st.session_state.frame_count = 0
        else:
            st.session_state.current_command = COMMAND_NONE

    st.divider()

    # Camera index selector
    cam_index = st.number_input(
        "Camera index", min_value=0, max_value=5, value=0, step=1,
        help="0 = built-in webcam. Try 1 or 2 for external cameras."
    )

    # Detection confidence slider
    det_conf = st.slider(
        "Detection confidence", 0.4, 1.0, 0.70, 0.05,
        help="Higher = fewer false positives but harder to trigger."
    )

    st.divider()
    st.markdown("### 🗺️ Gesture Map")

    gesture_table = {
        "👈 Point LEFT":   "Move LEFT",
        "👉 Point RIGHT":  "Move RIGHT",
        "👍 Thumb UP":     "Move FORWARD",
        "👎 Thumb DOWN":   "Move BACKWARD",
        "✋ Open Palm":    "STOP",
    }
    for gesture, cmd in gesture_table.items():
        st.markdown(f"**{gesture}** → `{cmd}`")

    st.divider()

    # FPS display (updated each rerun)
    if st.session_state.camera_running and st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        fps     = st.session_state.frame_count / max(elapsed, 1e-3)
        st.caption(f"⚡ FPS: {fps:.1f}  |  Frames: {st.session_state.frame_count}")

    st.caption("Built with MediaPipe · OpenCV · Streamlit")


# ─────────────────────────────────────────────
#  Main content area
# ─────────────────────────────────────────────

# Title row
st.markdown('<p class="main-title">🤖 AI Gesture Controlled Robot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Control a virtual robot using hand gestures via your webcam</p>', unsafe_allow_html=True)

# Layout: video feed (left) | command panel (right)
col_video, col_panel = st.columns([3, 1], gap="large")

with col_panel:
    # Status indicator
    if st.session_state.camera_running:
        st.markdown('<span class="status-pill status-active">● LIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-inactive">● OFFLINE</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Current command card placeholder
    command_placeholder = st.empty()

    def render_command_card(command: str):
        command_placeholder.markdown(
            f"""
            <div class="command-card">
                <div class="command-label">Robot Command</div>
                <div class="command-text">{command}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_command_card(st.session_state.current_command)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mini robot visualisation (ASCII art block)
    robot_ascii = st.empty()

    def update_robot_art(command: str):
        art_map = {
            COMMAND_FORWARD:  "```\n  ↑↑↑\n [🤖]\n  | |\n```",
            COMMAND_BACKWARD: "```\n  | |\n [🤖]\n  ↓↓↓\n```",
            COMMAND_LEFT:     "```\n←← [🤖]\n    | |\n```",
            COMMAND_RIGHT:    "```\n    [🤖] →→\n     | |\n```",
            COMMAND_STOP:     "```\n  ███\n [🤖]\n  | |\n```",
        }
        art = art_map.get(command, "```\n [🤖]\n  | |\n```")
        robot_ascii.markdown(art)

    update_robot_art(st.session_state.current_command)

# Video placeholder in left column
with col_video:
    if not st.session_state.camera_running:
        st.info("👆 Click **▶ Start Camera** in the sidebar to begin gesture detection.", icon="📷")
    video_placeholder = st.empty()


# ─────────────────────────────────────────────
#  Main camera loop
# ─────────────────────────────────────────────

if st.session_state.camera_running:
    cap      = cv2.VideoCapture(int(cam_index))
    detector = GestureDetector(
        min_detection_confidence=float(det_conf),
        min_tracking_confidence=0.6,
    )

    if not cap.isOpened():
        st.error(
            f"❌ Could not open camera index {int(cam_index)}. "
            "Try a different index in the sidebar or check your webcam connection.",
            icon="🚫",
        )
        st.session_state.camera_running = False
        st.stop()

    try:
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read frame. Camera may have disconnected.")
                break

            # ── Process frame ───────────────────────────
            result, annotated = detector.process_frame(frame)

            # Draw status overlay onto annotated frame
            annotated = draw_command_overlay(annotated, result)

            # ── Update UI ───────────────────────────────
            st.session_state.current_command = result.command
            st.session_state.frame_count    += 1

            # Convert BGR → RGB for Streamlit
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Refresh command card and robot art
            render_command_card(result.command)
            update_robot_art(result.command)

    finally:
        cap.release()
        detector.release()