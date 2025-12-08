"""
Data Collection Tool for Hand Gesture Recognition
Phase 4: Collect labeled landmark data for ML training
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from gesture_recognition import GestureRecognizer, GESTURES, HandLandmark

# Page config
st.set_page_config(
    page_title="Gesture Data Collection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    }
    
    .recording-active {
        color: #ff4444;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .gesture-btn {
        font-size: 1.5em;
        padding: 10px 20px;
        margin: 5px;
    }
    
    .progress-good { color: #00ff88; }
    .progress-medium { color: #ffaa00; }
    .progress-low { color: #ff4444; }
</style>
""", unsafe_allow_html=True)

# Constants
DATA_DIR = Path("data")
SESSIONS_DIR = DATA_DIR / "sessions"
TARGET_SAMPLES_PER_GESTURE = 150  # Target for good training data

# All gestures we want to collect data for
COLLECTION_GESTURES = [
    "thumbs_up", "thumbs_down", "raised_hand", "clapping",
    "peace", "ok_sign", "pointing", "fist", "rock_on",
    "none"
]


def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "combined").mkdir(exist_ok=True)


def get_existing_sessions():
    """Get list of existing session directories."""
    if not SESSIONS_DIR.exists():
        return []
    return [d.name for d in SESSIONS_DIR.iterdir() if d.is_dir()]


def load_session_metadata(session_name: str) -> dict:
    """Load metadata for a session."""
    metadata_path = SESSIONS_DIR / session_name / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def save_session_metadata(session_name: str, metadata: dict):
    """Save metadata for a session."""
    session_dir = SESSIONS_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = session_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_session_sample_counts(session_name: str) -> dict:
    """Count samples per gesture in a session."""
    session_dir = SESSIONS_DIR / session_name
    counts = {}
    
    for gesture in COLLECTION_GESTURES:
        gesture_dir = session_dir / gesture
        if gesture_dir.exists():
            csv_files = list(gesture_dir.glob("*.csv"))
            # Count total rows across all CSVs (excluding header)
            total_samples = 0
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    total_samples += len(df)
                except:
                    pass
            counts[gesture] = total_samples
        else:
            counts[gesture] = 0
    
    return counts


def get_aggregate_statistics() -> dict:
    """Get comprehensive statistics across all sessions."""
    sessions = get_existing_sessions()
    
    stats = {
        "total_sessions": len(sessions),
        "total_samples": 0,
        "participants": set(),
        "hand_distribution": {"left": 0, "right": 0, "both": 0, "unknown": 0},
        "gesture_counts": {g: 0 for g in COLLECTION_GESTURES},
        "per_participant": {},
        "per_session": {},
        "clips_per_gesture": {g: 0 for g in COLLECTION_GESTURES},
    }
    
    for session_name in sessions:
        metadata = load_session_metadata(session_name)
        session_counts = get_session_sample_counts(session_name)
        
        # Participant tracking
        participant = metadata.get("participant", "Unknown")
        stats["participants"].add(participant)
        
        if participant not in stats["per_participant"]:
            stats["per_participant"][participant] = {
                "sessions": 0,
                "samples": 0,
                "gestures": {g: 0 for g in COLLECTION_GESTURES}
            }
        stats["per_participant"][participant]["sessions"] += 1
        
        # Hand distribution
        hand = metadata.get("hand", "unknown")
        if hand in stats["hand_distribution"]:
            stats["hand_distribution"][hand] += 1
        else:
            stats["hand_distribution"]["unknown"] += 1
        
        # Session stats
        session_total = sum(session_counts.values())
        stats["per_session"][session_name] = {
            "participant": participant,
            "hand": hand,
            "samples": session_total,
            "gestures": session_counts
        }
        
        # Aggregate counts
        stats["total_samples"] += session_total
        stats["per_participant"][participant]["samples"] += session_total
        
        for gesture, count in session_counts.items():
            stats["gesture_counts"][gesture] += count
            stats["per_participant"][participant]["gestures"][gesture] += count
        
        # Count clips per gesture
        session_dir = SESSIONS_DIR / session_name
        for gesture in COLLECTION_GESTURES:
            gesture_dir = session_dir / gesture
            if gesture_dir.exists():
                stats["clips_per_gesture"][gesture] += len(list(gesture_dir.glob("*.csv")))
    
    stats["participants"] = list(stats["participants"])
    stats["num_participants"] = len(stats["participants"])
    
    return stats


def normalize_landmarks(hand_landmarks) -> list:
    """
    Normalize landmarks relative to wrist and scale to unit distance.
    Returns flat list of [x1, y1, z1, x2, y2, z2, ...] for 21 landmarks.
    """
    # Get wrist position as origin
    wrist = hand_landmarks.landmark[HandLandmark.WRIST]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
    
    # Collect all landmarks relative to wrist
    points = []
    for lm in hand_landmarks.landmark:
        points.append([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
    
    points = np.array(points)
    
    # Scale so max distance from wrist is 1
    distances = np.sqrt(np.sum(points**2, axis=1))
    max_dist = np.max(distances)
    if max_dist > 0:
        points = points / max_dist
    
    # Flatten to 1D list
    return points.flatten().tolist()


def save_recording(session_name: str, gesture: str, frames_data: list):
    """Save recorded frames to CSV."""
    if not frames_data:
        return None
    
    session_dir = SESSIONS_DIR / session_name
    gesture_dir = session_dir / gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_clips = list(gesture_dir.glob("clip_*.csv"))
    clip_num = len(existing_clips) + 1
    filename = f"clip_{clip_num:03d}_{timestamp}.csv"
    filepath = gesture_dir / filename
    
    # Create column names for landmarks
    columns = ["frame_idx", "timestamp"]
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")
    
    # Create DataFrame
    rows = []
    for frame_idx, (timestamp, landmarks) in enumerate(frames_data):
        row = [frame_idx, timestamp] + landmarks + [gesture]
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(filepath, index=False)
    
    return filepath, len(frames_data)


def export_combined_dataset(session_names: list = None):
    """Export all sessions to a combined CSV for training."""
    if session_names is None:
        session_names = get_existing_sessions()
    
    all_data = []
    
    for session_name in session_names:
        session_dir = SESSIONS_DIR / session_name
        for gesture in COLLECTION_GESTURES:
            gesture_dir = session_dir / gesture
            if gesture_dir.exists():
                for csv_file in gesture_dir.glob("*.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        df["session"] = session_name
                        all_data.append(df)
                    except Exception as e:
                        st.warning(f"Error reading {csv_file}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = DATA_DIR / "combined" / "all_landmarks.csv"
        combined.to_csv(output_path, index=False)
        return output_path, len(combined)
    
    return None, 0


def init_session_state():
    """Initialize session state variables."""
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = None
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'current_gesture' not in st.session_state:
        st.session_state.current_gesture = None
    if 'frames_buffer' not in st.session_state:
        st.session_state.frames_buffer = []
    if 'session_name' not in st.session_state:
        st.session_state.session_name = None
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False


def get_recognizer():
    """Get or create gesture recognizer."""
    if st.session_state.recognizer is None:
        st.session_state.recognizer = GestureRecognizer(smoothing_window=5)
    return st.session_state.recognizer


def main():
    """Main data collection application."""
    init_session_state()
    ensure_directories()
    
    # Header
    st.title("📊 Gesture Data Collection Tool")
    st.markdown("*Collect labeled hand landmark data for ML training*")
    
    # Sidebar - Session Management
    with st.sidebar:
        st.header("📁 Session Management")
        
        # Session selection/creation
        existing_sessions = get_existing_sessions()
        
        session_mode = st.radio(
            "Session",
            ["Create New", "Load Existing"],
            horizontal=True
        )
        
        if session_mode == "Create New":
            col1, col2 = st.columns(2)
            with col1:
                participant = st.text_input("Participant Name", value="")
            with col2:
                hand_pref = st.selectbox("Primary Hand", ["right", "left", "both"])
            
            lighting = st.selectbox("Lighting", ["natural", "bright", "dim", "mixed"])
            distance = st.selectbox("Distance", ["close", "medium", "far", "varied"])
            
            if st.button("🆕 Create Session", use_container_width=True):
                if participant:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_name = f"session_{timestamp}_{participant.lower().replace(' ', '_')}"
                    
                    metadata = {
                        "session_id": session_name,
                        "participant": participant,
                        "created": datetime.now().isoformat(),
                        "hand": hand_pref,
                        "lighting": lighting,
                        "distance": distance,
                        "samples_collected": {}
                    }
                    
                    save_session_metadata(session_name, metadata)
                    st.session_state.session_name = session_name
                    st.success(f"✅ Created session: {session_name}")
                    st.rerun()
                else:
                    st.error("Please enter participant name")
        
        else:  # Load Existing
            if existing_sessions:
                selected_session = st.selectbox(
                    "Select Session",
                    existing_sessions,
                    format_func=lambda x: x.replace("session_", "").replace("_", " ")
                )
                
                if st.button("📂 Load Session", use_container_width=True):
                    st.session_state.session_name = selected_session
                    st.success(f"✅ Loaded session: {selected_session}")
                    st.rerun()
            else:
                st.info("No existing sessions. Create a new one.")
        
        st.divider()
        
        # Current session info
        if st.session_state.session_name:
            st.subheader("📋 Current Session")
            metadata = load_session_metadata(st.session_state.session_name)
            if metadata:
                st.write(f"**Participant:** {metadata.get('participant', 'Unknown')}")
                st.write(f"**Hand:** {metadata.get('hand', 'Unknown')}")
                st.write(f"**Lighting:** {metadata.get('lighting', 'Unknown')}")
        
        st.divider()
        
        # Export options
        st.subheader("📤 Export Data")
        if st.button("🔄 Export Combined CSV", use_container_width=True):
            filepath, count = export_combined_dataset()
            if filepath:
                st.success(f"✅ Exported {count} samples to {filepath}")
            else:
                st.warning("No data to export")
    
    # Main content - Tabs
    tab_record, tab_dashboard = st.tabs(["📹 Record Data", "📊 Dashboard"])
    
    # ==================== DASHBOARD TAB ====================
    with tab_dashboard:
        st.subheader("📊 Aggregate Statistics Dashboard")
        
        stats = get_aggregate_statistics()
        
        if stats["total_sessions"] == 0:
            st.info("No data collected yet. Create a session and start recording!")
        else:
            # Top-level metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total Sessions", stats["total_sessions"])
            with metric_cols[1]:
                st.metric("Total Samples", f"{stats['total_samples']:,}")
            with metric_cols[2]:
                st.metric("Participants", stats["num_participants"])
            with metric_cols[3]:
                avg_per_gesture = stats["total_samples"] // len(COLLECTION_GESTURES) if COLLECTION_GESTURES else 0
                st.metric("Avg per Gesture", avg_per_gesture)
            
            st.divider()
            
            # Per-gesture breakdown
            dash_col1, dash_col2 = st.columns(2)
            
            with dash_col1:
                st.markdown("### 🎯 Samples per Gesture")
                for gesture in COLLECTION_GESTURES:
                    count = stats["gesture_counts"].get(gesture, 0)
                    clips = stats["clips_per_gesture"].get(gesture, 0)
                    gesture_info = GESTURES.get(gesture, GESTURES["none"])
                    
                    # Progress towards target
                    target = TARGET_SAMPLES_PER_GESTURE * stats["num_participants"] if stats["num_participants"] > 0 else TARGET_SAMPLES_PER_GESTURE
                    progress = min(count / target, 1.0) if target > 0 else 0
                    
                    if count >= target:
                        icon = "✅"
                    elif count >= target // 2:
                        icon = "🔶"
                    else:
                        icon = "⬜"
                    
                    st.progress(progress, text=f"{icon} {gesture_info.emoji} {gesture_info.name}: {count:,} samples ({clips} clips)")
            
            with dash_col2:
                st.markdown("### 👥 Per Participant")
                for participant, pdata in stats["per_participant"].items():
                    with st.expander(f"**{participant}** - {pdata['samples']:,} samples ({pdata['sessions']} sessions)"):
                        for gesture in COLLECTION_GESTURES:
                            g_count = pdata["gestures"].get(gesture, 0)
                            gesture_info = GESTURES.get(gesture, GESTURES["none"])
                            st.write(f"{gesture_info.emoji} {gesture_info.name}: {g_count}")
            
            st.divider()
            
            # Hand distribution
            hand_col1, hand_col2 = st.columns(2)
            
            with hand_col1:
                st.markdown("### ✋ Hand Distribution (by session)")
                hand_dist = stats["hand_distribution"]
                total_hand = sum(hand_dist.values())
                if total_hand > 0:
                    for hand, count in hand_dist.items():
                        if count > 0:
                            pct = (count / total_hand) * 100
                            st.write(f"**{hand.capitalize()}:** {count} sessions ({pct:.0f}%)")
                else:
                    st.write("No data yet")
            
            with hand_col2:
                st.markdown("### 📋 Session Details")
                session_data = []
                for session_name, sdata in stats["per_session"].items():
                    session_data.append({
                        "Session": session_name.replace("session_", "")[:20],
                        "Participant": sdata["participant"],
                        "Hand": sdata["hand"],
                        "Samples": sdata["samples"]
                    })
                if session_data:
                    st.dataframe(pd.DataFrame(session_data), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Data quality check
            st.markdown("### ⚠️ Data Quality Check")
            issues = []
            
            # Check for imbalanced classes
            counts = list(stats["gesture_counts"].values())
            if counts and max(counts) > 0:
                min_count = min(counts)
                max_count = max(counts)
                if max_count > min_count * 3:
                    issues.append(f"⚠️ **Class imbalance:** Most samples = {max_count:,}, fewest = {min_count:,}")
            
            # Check for missing gestures
            missing = [g for g, c in stats["gesture_counts"].items() if c == 0]
            if missing:
                missing_names = [GESTURES.get(g, GESTURES["none"]).name for g in missing]
                issues.append(f"⚠️ **Missing gestures:** {', '.join(missing_names)}")
            
            # Check for low sample counts
            low_count = [g for g, c in stats["gesture_counts"].items() if 0 < c < 50]
            if low_count:
                low_names = [f"{GESTURES.get(g, GESTURES['none']).name} ({stats['gesture_counts'][g]})" for g in low_count]
                issues.append(f"⚠️ **Low samples (<50):** {', '.join(low_names)}")
            
            # Check for single participant
            if stats["num_participants"] == 1:
                issues.append("⚠️ **Single participant:** Add more participants for better generalization")
            
            # Check hand distribution
            if stats["hand_distribution"]["left"] == 0 or stats["hand_distribution"]["right"] == 0:
                issues.append("⚠️ **Hand coverage:** Missing data for left or right hand")
            
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("✅ Data looks good! No major issues detected.")
    
    # ==================== RECORD TAB ====================
    with tab_record:
        if not st.session_state.session_name:
            st.info("👈 Create or load a session in the sidebar to start collecting data")
        else:
            # Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📹 Recording Interface")
                
                # Camera controls
                cam_col1, cam_col2 = st.columns(2)
                with cam_col1:
                    start_cam = st.button("▶️ Start Camera", use_container_width=True, type="primary")
                with cam_col2:
                    stop_cam = st.button("⏹️ Stop Camera", use_container_width=True)
                
                video_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Recording controls
                st.subheader("🎯 Select Gesture & Record")
                
                # Gesture selection buttons
                gesture_cols = st.columns(5)
                
                for idx, gesture in enumerate(COLLECTION_GESTURES):
                    g_col = gesture_cols[idx % 5]
                    gesture_info = GESTURES.get(gesture, GESTURES["none"])
                    btn_label = f"{gesture_info.emoji} {gesture_info.name}" if gesture_info.emoji else gesture_info.name
                    
                    with g_col:
                        if st.button(btn_label, key=f"btn_{gesture}", use_container_width=True):
                            st.session_state.current_gesture = gesture
                
                # Recording status
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                with rec_col1:
                    if st.session_state.current_gesture:
                        gesture_info = GESTURES.get(st.session_state.current_gesture, GESTURES["none"])
                        st.info(f"**Selected:** {gesture_info.emoji} {gesture_info.name}")
                    else:
                        st.warning("Select a gesture above")
                
                with rec_col2:
                    record_btn = st.button(
                        "🔴 START Recording" if not st.session_state.recording else "⏹️ STOP Recording",
                        use_container_width=True,
                        type="primary" if not st.session_state.recording else "secondary",
                        disabled=st.session_state.current_gesture is None
                    )
                    
                    if record_btn:
                        st.session_state.recording = not st.session_state.recording
                        if st.session_state.recording:
                            st.session_state.frames_buffer = []
                        else:
                            # Save recording
                            if st.session_state.frames_buffer:
                                filepath, count = save_recording(
                                    st.session_state.session_name,
                                    st.session_state.current_gesture,
                                    st.session_state.frames_buffer
                                )
                                if filepath:
                                    st.success(f"✅ Saved {count} frames to {filepath}")
                            st.session_state.frames_buffer = []
                
                with rec_col3:
                    if st.session_state.recording:
                        st.markdown(f"<p class='recording-active'>🔴 RECORDING... ({len(st.session_state.frames_buffer)} frames)</p>", 
                                   unsafe_allow_html=True)
                    else:
                        st.write("Ready to record")
            
            with col2:
                st.subheader("📊 Session Progress")
                
                # Progress for current session
                counts = get_session_sample_counts(st.session_state.session_name)
                
                total_samples = sum(counts.values())
                st.metric("Total Samples", total_samples)
                
                st.markdown("**Per Gesture:**")
                for gesture in COLLECTION_GESTURES:
                    count = counts.get(gesture, 0)
                    gesture_info = GESTURES.get(gesture, GESTURES["none"])
                    
                    # Color based on progress
                    if count >= TARGET_SAMPLES_PER_GESTURE:
                        icon = "✅"
                    elif count >= TARGET_SAMPLES_PER_GESTURE // 2:
                        icon = "🔶"
                    else:
                        icon = "⬜"
                    
                    progress = min(count / TARGET_SAMPLES_PER_GESTURE, 1.0)
                    st.progress(progress, text=f"{icon} {gesture_info.name}: {count}/{TARGET_SAMPLES_PER_GESTURE}")
                
                st.divider()
                
                # Quick stats
                st.subheader("📈 Session Stats")
                metadata = load_session_metadata(st.session_state.session_name)
                if metadata:
                    created = metadata.get('created', 'Unknown')
                    if created != 'Unknown':
                        try:
                            created_dt = datetime.fromisoformat(created)
                            st.write(f"**Created:** {created_dt.strftime('%Y-%m-%d %H:%M')}")
                        except:
                            pass
            
            # Camera loop (outside columns but inside tab_record and else block)
            if start_cam:
                st.session_state.camera_running = True
            if stop_cam:
                st.session_state.camera_running = False
                if st.session_state.recognizer:
                    st.session_state.recognizer.close()
                    st.session_state.recognizer = None
            
            if st.session_state.camera_running:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    status_placeholder.error("❌ Could not open webcam")
                    st.session_state.camera_running = False
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    recognizer = get_recognizer()
                    status_placeholder.success("✅ Camera running")
                    
                    try:
                        while st.session_state.camera_running:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Mirror
                            frame = cv2.flip(frame, 1)
                            
                            # Process for landmarks
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = recognizer.hands.process(rgb_frame)
                            
                            # Draw landmarks
                            if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    recognizer.mp_drawing.draw_landmarks(
                                        frame,
                                        hand_landmarks,
                                        recognizer.mp_hands.HAND_CONNECTIONS
                                    )
                                    
                                    # If recording, save normalized landmarks
                                    if st.session_state.recording and st.session_state.current_gesture:
                                        normalized = normalize_landmarks(hand_landmarks)
                                        timestamp = time.time()
                                        st.session_state.frames_buffer.append((timestamp, normalized))
                            
                            # Draw recording indicator
                            if st.session_state.recording:
                                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
                                cv2.putText(frame, f"REC: {len(st.session_state.frames_buffer)}", 
                                           (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                
                                # Show current gesture being recorded
                                if st.session_state.current_gesture:
                                    gesture_info = GESTURES.get(st.session_state.current_gesture, GESTURES["none"])
                                    cv2.putText(frame, f"Gesture: {gesture_info.name}", 
                                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
                            # Show hand detection status
                            hand_status = "Hand Detected" if results.multi_hand_landmarks else "No Hand"
                            status_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
                            cv2.putText(frame, hand_status, (10, frame.shape[0] - 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                            
                            # Display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                            
                            time.sleep(0.01)
                            
                    finally:
                        cap.release()
                        status_placeholder.info("📷 Camera stopped")
            else:
                video_placeholder.info("👆 Click **Start Camera** to begin")
    
    # Guidelines
    st.divider()
    with st.expander("📋 **Data Collection Guidelines** (Click to expand)", expanded=False):
        guide_col1, guide_col2 = st.columns(2)
        
        with guide_col1:
            st.markdown("""
            ### ⏱️ Recording Duration
            - **Per clip:** 3-5 seconds of steady gesture
            - **Total per gesture:** Aim for 150+ samples (frames)
            - **Multiple clips:** Better to do 5-10 short clips than 1 long one
            
            ### ✋ Hand Usage
            - **One hand at a time** for single-hand gestures
            - **Record both hands separately** (do all gestures with right, then left)
            - **Clapping:** Use both hands together
            
            ### 📏 Distance & Position
            - **Vary distance:** Some clips close, some at arm's length
            - **Keep hand in frame:** Full hand should be visible
            - **Center-ish:** Hand roughly in middle of frame
            """)
        
        with guide_col2:
            st.markdown("""
            ### 💡 Lighting & Background
            - **Good lighting:** Face a window or lamp
            - **Avoid backlight:** Don't sit with window behind you
            - **Plain background:** Helps with hand detection
            
            ### 🎯 Quality Tips
            - **Hold steady:** Don't move while recording
            - **Clear gesture:** Make the gesture obvious/exaggerated
            - **Wait for detection:** Green "Hand Detected" before recording
            - **Natural variation:** Slight angle/position changes between clips
            
            ### ⚠️ Common Mistakes
            - ❌ Recording while moving hand into position
            - ❌ Fingers partially out of frame
            - ❌ Poor lighting causing detection drops
            - ❌ Recording "none" class without removing hand
            """)
        
        st.info("""
        **Workflow:** Select gesture → Position hand → Wait for "Hand Detected" → Start Recording → 
        Hold 3-5 seconds → Stop Recording → Repeat with slight variation
        """)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #888; margin-top: 20px;'>
        <p>📊 <b>Gesture Data Collection Tool</b> | Phase 4</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

