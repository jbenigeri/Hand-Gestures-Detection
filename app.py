"""
Streamlit UI for Hand Gesture Recognition
Phase 2: Interactive web-based demo with controls and statistics
"""

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime

# Import from gesture_recognition module
from gesture_recognition import (
    GestureRecognizer, 
    ReactionOverlay,
    GESTURES, 
    HandLandmark,
    put_text_with_emoji
)

# Page config
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .stTitle {
        color: #e94560 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(233, 69, 96, 0.1);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Stats box */
    .stats-box {
        background: rgba(15, 52, 96, 0.5);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Reaction log */
    .reaction-log {
        background: rgba(26, 26, 46, 0.8);
        border-left: 3px solid #e94560;
        padding: 10px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        font-family: 'Fira Code', monospace;
    }
    
    /* Toggle labels */
    .gesture-toggle {
        font-size: 1.2em;
        padding: 5px;
    }
    
    /* Status indicator */
    .status-active {
        color: #00ff88;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = None
    if 'overlay' not in st.session_state:
        st.session_state.overlay = ReactionOverlay()
    if 'reaction_history' not in st.session_state:
        st.session_state.reaction_history = deque(maxlen=20)
    if 'gesture_counts' not in st.session_state:
        st.session_state.gesture_counts = {g: 0 for g in GESTURES.keys() if g != "none"}
    if 'total_frames' not in st.session_state:
        st.session_state.total_frames = 0
    if 'fps_history' not in st.session_state:
        st.session_state.fps_history = deque(maxlen=30)
    if 'last_gesture' not in st.session_state:
        st.session_state.last_gesture = "none"
    if 'last_reaction_time' not in st.session_state:
        st.session_state.last_reaction_time = 0
    if 'running' not in st.session_state:
        st.session_state.running = False


def get_recognizer(smoothing_window: int, detection_confidence: float, use_ml: bool = False, ml_confidence: float = 0.6):
    """Get or create gesture recognizer with current settings."""
    if st.session_state.recognizer is None:
        st.session_state.recognizer = GestureRecognizer(smoothing_window=smoothing_window, use_ml=use_ml)
        # Update confidence settings
        st.session_state.recognizer.hands.close()
        st.session_state.recognizer.hands = st.session_state.recognizer.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5,
        )
    
    # Update ML settings
    st.session_state.recognizer.use_ml = use_ml
    st.session_state.recognizer.ml_confidence_threshold = ml_confidence
    
    # Try to load ML model if ML mode enabled
    if use_ml and st.session_state.recognizer.ml_model is None:
        st.session_state.recognizer.load_ml_model()
    
    return st.session_state.recognizer


def draw_frame_overlay(frame: np.ndarray, gesture: str, debug_mode: bool, 
                       results, recognizer: GestureRecognizer,
                       overlay: ReactionOverlay, trigger_reaction: bool = False,
                       confidence: float = 1.0, show_confidence: bool = False,
                       use_ml: bool = False) -> np.ndarray:
    """Draw overlays on the frame including floating emoji reactions."""
    h, w = frame.shape[:2]
    
    # Draw landmarks in debug mode
    if debug_mode and results and results.multi_hand_landmarks:
        frame = recognizer.draw_landmarks(frame, results)
        # Debug mode indicator
        cv2.putText(frame, "DEBUG MODE", (w - 180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw ML mode indicator
    if use_ml:
        mode_text = "ML MODE"
        mode_color = (0, 255, 0) if recognizer.ml_model is not None else (0, 0, 255)
        cv2.putText(frame, mode_text, (w - 180, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    # Draw confidence if enabled
    if show_confidence and gesture != "none":
        conf_text = f"Conf: {confidence:.0%}"
        cv2.putText(frame, conf_text, (w - 180, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Trigger floating reaction if needed
    if trigger_reaction and gesture != "none":
        overlay.trigger_reaction(gesture, w, h)
    
    # Draw the full overlay (floating emoji + status bar + log)
    frame = overlay.draw_overlay(frame, gesture)
    
    return frame


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Header
    st.title("🖐️ Hand Gesture Recognition")
    st.markdown("*Real-time gesture detection for video call reactions*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Detection settings
        st.subheader("Detection")
        detection_confidence = st.slider(
            "Detection Confidence",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values = more accurate but may miss some hands"
        )
        
        smoothing_window = st.slider(
            "Smoothing Window",
            min_value=3,
            max_value=15,
            value=7,
            step=2,
            help="Number of frames for temporal smoothing (higher = more stable)"
        )
        
        reaction_cooldown = st.slider(
            "Reaction Cooldown (s)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.25,
            help="Minimum time between reaction triggers"
        )
        
        st.divider()
        
        # Gesture toggles - Core gestures
        st.subheader("🎯 Core Gestures")
        gesture_enabled = {}
        
        # Core gestures (Phase 1) - ON by default
        core_gestures = ["thumbs_up", "thumbs_down", "raised_hand", "clapping"]
        for gesture_key in core_gestures:
            gesture_info = GESTURES[gesture_key]
            gesture_enabled[gesture_key] = st.checkbox(
                f"{gesture_info.emoji} {gesture_info.name}",
                value=True,
                key=f"toggle_{gesture_key}"
            )
        
        st.divider()
        
        # Extended gestures (Phase 3) - OFF by default
        st.subheader("🔮 Extended Gestures")
        st.caption("*Toggle on to enable*")
        
        # Finger counting group
        with st.expander("🔢 Finger Counting", expanded=False):
            for i in range(1, 6):
                gesture_key = f"finger_count_{i}"
                gesture_info = GESTURES[gesture_key]
                gesture_enabled[gesture_key] = st.checkbox(
                    f"{gesture_info.emoji} {gesture_info.name}",
                    value=False,
                    key=f"toggle_{gesture_key}"
                )
        
        # Other extended gestures
        extended_gestures = ["peace", "ok_sign", "pointing", "fist", "rock_on"]
        for gesture_key in extended_gestures:
            gesture_info = GESTURES[gesture_key]
            gesture_enabled[gesture_key] = st.checkbox(
                f"{gesture_info.emoji} {gesture_info.name}",
                value=False,
                key=f"toggle_{gesture_key}"
            )
        
        st.divider()
        
        # ML Mode
        st.subheader("🧠 ML Mode")
        use_ml = st.checkbox(
            "Use ML Classifier",
            value=False,
            help="Use trained ML model instead of heuristics (requires trained model in models/active/)"
        )
        
        if use_ml:
            ml_confidence = st.slider(
                "Confidence Threshold",
                min_value=0.3,
                max_value=0.95,
                value=0.6,
                step=0.05,
                help="Minimum confidence for ML predictions (lower = more detections, higher = more accurate)"
            )
            st.caption("Falls back to heuristics if confidence below threshold")
        else:
            ml_confidence = 0.6
        
        st.divider()
        
        # Debug mode
        st.subheader("🔧 Debug")
        debug_mode = st.checkbox("Show Hand Landmarks", value=False)
        show_confidence = st.checkbox("Show Confidence", value=False, help="Display ML confidence scores")
        
        st.divider()
        
        # Reset stats button
        if st.button("🔄 Reset Statistics", use_container_width=True):
            st.session_state.gesture_counts = {g: 0 for g in GESTURES.keys() if g != "none"}
            st.session_state.total_frames = 0
            st.session_state.reaction_history.clear()
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        
        # Camera controls
        run_col1, run_col2 = st.columns(2)
        with run_col1:
            start_button = st.button("▶️ Start Camera", use_container_width=True, type="primary")
        with run_col2:
            stop_button = st.button("⏹️ Stop Camera", use_container_width=True)
        
        # Video placeholder
        video_placeholder = st.empty()
        
        # Status
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Statistics")
        
        # Metrics
        metrics_cols = st.columns(2)
        fps_metric = metrics_cols[0].empty()
        frames_metric = metrics_cols[1].empty()
        
        # Gesture counts
        st.markdown("**Gesture Counts**")
        counts_placeholder = st.empty()
        
        st.divider()
        
        # Reaction history
        st.subheader("📜 Reaction History")
        history_placeholder = st.empty()
    
    # Handle start/stop
    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False
        if st.session_state.recognizer:
            st.session_state.recognizer.close()
            st.session_state.recognizer = None
    
    # Main loop
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            status_placeholder.error("❌ Could not open webcam. Please check your camera connection.")
            st.session_state.running = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            recognizer = get_recognizer(smoothing_window, detection_confidence, use_ml, ml_confidence)
            
            # Show ML status
            if use_ml:
                if recognizer.ml_model is not None:
                    status_placeholder.success("✅ Camera running with ML model...")
                else:
                    status_placeholder.warning("⚠️ Camera running (ML model not found, using heuristics)")
            else:
                status_placeholder.success("✅ Camera running...")
            
            try:
                while st.session_state.running:
                    frame_start = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.error("❌ Failed to read frame")
                        break
                    
                    # Mirror the frame
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame with enabled gestures filter and ML mode
                    gesture, confidence, results = recognizer.process_frame(frame, gesture_enabled, use_ml)
                    
                    # Update statistics
                    st.session_state.total_frames += 1
                    
                    # Track FPS
                    frame_time = time.time() - frame_start
                    fps = 1.0 / frame_time if frame_time > 0 else 0
                    st.session_state.fps_history.append(fps)
                    avg_fps = sum(st.session_state.fps_history) / len(st.session_state.fps_history)
                    
                    # Check if we should trigger a new reaction
                    trigger_reaction = False
                    current_time = time.time()
                    if gesture != "none" and gesture != st.session_state.last_gesture:
                        if current_time - st.session_state.last_reaction_time > reaction_cooldown:
                            trigger_reaction = True
                            
                            # Update counts
                            st.session_state.gesture_counts[gesture] += 1
                            
                            # Add to history
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            gesture_info = GESTURES[gesture]
                            st.session_state.reaction_history.appendleft(
                                f"{gesture_info.emoji} {gesture_info.name} @ {timestamp}"
                            )
                            
                            st.session_state.last_reaction_time = current_time
                    
                    st.session_state.last_gesture = gesture
                    
                    # Draw overlays with floating emoji reactions
                    frame = draw_frame_overlay(frame, gesture, debug_mode, results, recognizer,
                                               st.session_state.overlay, trigger_reaction,
                                               confidence, show_confidence, use_ml)
                    
                    # Convert frame for display (BGR to RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Update metrics
                    fps_metric.metric("FPS", f"{avg_fps:.1f}")
                    frames_metric.metric("Frames", st.session_state.total_frames)
                    
                    # Update gesture counts display
                    counts_md = ""
                    for g, count in st.session_state.gesture_counts.items():
                        emoji = GESTURES[g].emoji
                        enabled = "✅" if gesture_enabled.get(g, True) else "❌"
                        counts_md += f"{enabled} {emoji} **{GESTURES[g].name}**: {count}\n\n"
                    counts_placeholder.markdown(counts_md)
                    
                    # Update history display
                    if st.session_state.reaction_history:
                        history_md = "\n\n".join([f"• {r}" for r in list(st.session_state.reaction_history)[:10]])
                        history_placeholder.markdown(history_md)
                    else:
                        history_placeholder.markdown("*No reactions yet...*")
                    
                    # Small delay to prevent overwhelming the UI
                    time.sleep(0.01)
                    
            finally:
                cap.release()
                status_placeholder.info("📷 Camera stopped")
    else:
        # Show placeholder when not running
        video_placeholder.info("👆 Click **Start Camera** to begin gesture recognition")
        
        # Show current stats
        fps_metric.metric("FPS", "—")
        frames_metric.metric("Frames", st.session_state.total_frames)
        
        counts_md = ""
        for g, count in st.session_state.gesture_counts.items():
            emoji = GESTURES[g].emoji
            enabled = "✅" if gesture_enabled.get(g, True) else "❌"
            counts_md += f"{enabled} {emoji} **{GESTURES[g].name}**: {count}\n\n"
        counts_placeholder.markdown(counts_md)
        
        if st.session_state.reaction_history:
            history_md = "\n\n".join([f"• {r}" for r in list(st.session_state.reaction_history)[:10]])
            history_placeholder.markdown(history_md)
        else:
            history_placeholder.markdown("*No reactions yet...*")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>🖐️ <b>Hand Gesture Recognition</b> | Phase 2: Streamlit UI</p>
        <p>Gestures: 👍 Thumbs Up | 👎 Thumbs Down | ✋ Raised Hand | 👏 Clapping</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

