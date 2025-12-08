"""
Real-Time Hand Gesture Recognition for Video Call Reactions
Using MediaPipe Hands + Heuristic Rules + ML Classifier

Supported Gestures:
- 👍 Thumbs Up
- 👎 Thumbs Down  
- ✋ Raised Hand (Open Palm)
- 👏 Clapping (two hands close together)
- Plus extended gestures with ML

Controls:
- Press 'q' to quit
- Press 'd' to toggle debug mode (show landmarks)
- Press 'm' to toggle ML mode (if model loaded)
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
from PIL import Image, ImageDraw, ImageFont
import platform
from pathlib import Path
import json

# Try to import ML dependencies (optional)
try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    joblib = None


# MediaPipe Hand Landmark indices
class HandLandmark:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class Gesture:
    name: str
    emoji: str
    color: Tuple[int, int, int]  # BGR
    extended: bool = False  # True for Phase 3 gestures (off by default)


# Gesture definitions
# Core gestures (Phase 1) - always enabled by default
GESTURES = {
    "thumbs_up": Gesture("Thumbs Up", "👍", (0, 200, 0)),
    "thumbs_down": Gesture("Thumbs Down", "👎", (0, 0, 200)),
    "raised_hand": Gesture("Raised Hand", "✋", (200, 150, 0)),
    "clapping": Gesture("Clapping", "👏", (200, 0, 200)),
    # Extended gestures (Phase 3) - off by default
    "finger_count_1": Gesture("One", "1️⃣", (255, 180, 0), extended=True),
    "finger_count_2": Gesture("Two", "2️⃣", (255, 160, 0), extended=True),
    "finger_count_3": Gesture("Three", "3️⃣", (255, 140, 0), extended=True),
    "finger_count_4": Gesture("Four", "4️⃣", (255, 120, 0), extended=True),
    "finger_count_5": Gesture("Five", "5️⃣", (255, 100, 0), extended=True),
    "peace": Gesture("Peace", "✌️", (0, 255, 200), extended=True),
    "ok_sign": Gesture("OK", "👌", (100, 255, 100), extended=True),
    "pointing": Gesture("Pointing", "👆", (255, 200, 0), extended=True),
    "fist": Gesture("Fist", "✊", (150, 100, 200), extended=True),
    "rock_on": Gesture("Rock On", "🤘", (255, 0, 150), extended=True),
    "none": Gesture("None", "", (128, 128, 128)),
}


def get_emoji_font(size: int = 32) -> ImageFont.FreeTypeFont:
    """Get a font that supports emoji rendering based on the platform."""
    system = platform.system()
    
    # Try platform-specific emoji fonts
    font_options = []
    if system == "Darwin":  # macOS
        font_options = [
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
    elif system == "Windows":
        font_options = [
            "C:/Windows/Fonts/seguiemj.ttf",  # Segoe UI Emoji
            "C:/Windows/Fonts/arial.ttf",
        ]
    else:  # Linux
        font_options = [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    
    for font_path in font_options:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    
    # Fallback to default font
    return ImageFont.load_default()


def put_text_with_emoji(frame: np.ndarray, text: str, position: Tuple[int, int],
                        font_size: int = 24, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Draw text with emoji support using PIL.
    
    Args:
        frame: OpenCV BGR image
        text: Text string (can include emoji)
        position: (x, y) position for the text
        font_size: Font size in pixels
        color: BGR color tuple
    
    Returns:
        Modified frame with text drawn
    """
    # Convert OpenCV BGR to PIL RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Get font with emoji support
    font = get_emoji_font(font_size)
    
    # Convert BGR to RGB for PIL
    rgb_color = (color[2], color[1], color[0])
    
    # Draw text
    draw.text(position, text, font=font, fill=rgb_color, embedded_color=True)
    
    # Convert back to OpenCV BGR
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return result


class GestureRecognizer:
    """
    Gesture recognition using MediaPipe hand landmarks.
    Supports both heuristic-based and ML-based classification.
    """
    
    def __init__(self, smoothing_window: int = 7, use_ml: bool = False, model_path: str = None):
        """
        Initialize the gesture recognizer.
        
        Args:
            smoothing_window: Number of frames for temporal smoothing
            use_ml: Whether to use ML classification (requires trained model)
            model_path: Path to model directory (default: models/active/)
        """
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        
        # Temporal smoothing
        self.gesture_history = deque(maxlen=smoothing_window)
        self.current_gesture = "none"
        
        # Clapping detection state
        self.prev_hand_distance = None
        self.clap_cooldown = 0
        
        # ML model state
        self.use_ml = use_ml and ML_AVAILABLE
        self.ml_model = None
        self.ml_scaler = None
        self.ml_label_encoder = None
        self.ml_confidence_threshold = 0.6
        self.ml_model_info = None
        
        # Load ML model if requested
        if self.use_ml:
            self.load_ml_model(model_path)
    
    def load_ml_model(self, model_path: str = None) -> bool:
        """
        Load a trained ML model for gesture classification.
        
        Args:
            model_path: Path to model directory. If None, uses models/active/
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not ML_AVAILABLE:
            print("Warning: joblib not installed. ML mode unavailable.")
            return False
        
        if model_path is None:
            model_path = Path("models") / "active"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"Warning: Model path not found: {model_path}")
            self.use_ml = False
            return False
        
        try:
            # Load model
            self.ml_model = joblib.load(model_path / "model.joblib")
            
            # Load label encoder
            self.ml_label_encoder = joblib.load(model_path / "label_encoder.joblib")
            
            # Load scaler if exists (for SVM)
            scaler_path = model_path / "scaler.joblib"
            if scaler_path.exists():
                self.ml_scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.ml_model_info = json.load(f)
            
            print(f"✅ Loaded ML model from {model_path}")
            if self.ml_model_info:
                print(f"   Model: {self.ml_model_info.get('model_name', 'unknown')}")
                print(f"   Accuracy: {self.ml_model_info.get('metrics', {}).get('accuracy', 0):.1%}")
            
            self.use_ml = True
            return True
            
        except Exception as e:
            print(f"Error loading ML model: {e}")
            self.use_ml = False
            return False
    
    def normalize_landmarks_for_ml(self, hand_landmarks) -> np.ndarray:
        """
        Normalize landmarks for ML model input.
        Same normalization as used during data collection.
        
        Returns:
            Flat array of [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
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
        
        # Flatten to 1D
        return points.flatten()
    
    def classify_gesture_ml(self, hand_landmarks) -> Tuple[str, float]:
        """
        Classify gesture using the ML model.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
        
        Returns:
            Tuple of (gesture_name, confidence)
        """
        if self.ml_model is None:
            return "none", 0.0
        
        # Normalize landmarks
        features = self.normalize_landmarks_for_ml(hand_landmarks)
        features = features.reshape(1, -1)
        
        # Scale if needed (SVM)
        if self.ml_scaler is not None:
            features = self.ml_scaler.transform(features)
        
        # Predict
        prediction = self.ml_model.predict(features)[0]
        
        # Get confidence (probability)
        confidence = 1.0
        if hasattr(self.ml_model, 'predict_proba'):
            probas = self.ml_model.predict_proba(features)[0]
            confidence = float(np.max(probas))
        
        # Decode label
        gesture_name = self.ml_label_encoder.inverse_transform([prediction])[0]
        
        return gesture_name, confidence
    
    def set_ml_mode(self, enabled: bool):
        """Enable or disable ML mode."""
        if enabled and not ML_AVAILABLE:
            print("Warning: joblib not installed. Cannot enable ML mode.")
            return
        
        if enabled and self.ml_model is None:
            # Try to load default model
            if not self.load_ml_model():
                print("Warning: No ML model loaded. Staying in heuristic mode.")
                return
        
        self.use_ml = enabled
        print(f"ML mode: {'ON' if enabled else 'OFF'}")
        
    def get_landmark_coords(self, hand_landmarks, idx: int) -> Tuple[float, float, float]:
        """Extract x, y, z coordinates for a landmark."""
        lm = hand_landmarks.landmark[idx]
        return lm.x, lm.y, lm.z
    
    def is_finger_extended(self, hand_landmarks, finger_tip: int, finger_pip: int, 
                           finger_mcp: int, is_thumb: bool = False) -> bool:
        """
        Check if a finger is extended using geometric heuristics.
        
        For fingers: tip should be above (lower y) the PIP joint
        For thumb: uses different logic based on hand orientation
        """
        tip = self.get_landmark_coords(hand_landmarks, finger_tip)
        pip = self.get_landmark_coords(hand_landmarks, finger_pip)
        mcp = self.get_landmark_coords(hand_landmarks, finger_mcp)
        
        if is_thumb:
            # Thumb extended if tip is far from palm center horizontally
            wrist = self.get_landmark_coords(hand_landmarks, HandLandmark.WRIST)
            index_mcp = self.get_landmark_coords(hand_landmarks, HandLandmark.INDEX_MCP)
            
            # Calculate thumb extension based on distance from wrist-index line
            thumb_dist = abs(tip[0] - wrist[0])
            return thumb_dist > 0.05
        else:
            # Finger is extended if tip is above PIP (smaller y = higher in image)
            return tip[1] < pip[1]
    
    def is_finger_curled(self, hand_landmarks, finger_tip: int, finger_mcp: int) -> bool:
        """Check if a finger is curled (tip below/close to MCP)."""
        tip = self.get_landmark_coords(hand_landmarks, finger_tip)
        mcp = self.get_landmark_coords(hand_landmarks, finger_mcp)
        
        # Finger curled if tip is below or at same level as MCP
        return tip[1] >= mcp[1] - 0.02
    
    def get_thumb_direction(self, hand_landmarks) -> str:
        """Determine if thumb is pointing up, down, or sideways."""
        thumb_tip = self.get_landmark_coords(hand_landmarks, HandLandmark.THUMB_TIP)
        thumb_ip = self.get_landmark_coords(hand_landmarks, HandLandmark.THUMB_IP)
        thumb_mcp = self.get_landmark_coords(hand_landmarks, HandLandmark.THUMB_MCP)
        wrist = self.get_landmark_coords(hand_landmarks, HandLandmark.WRIST)
        
        # Vertical difference between thumb tip and base
        vertical_diff = thumb_tip[1] - thumb_mcp[1]
        
        # Threshold for determining direction
        if vertical_diff < -0.08:  # Tip is significantly above base
            return "up"
        elif vertical_diff > 0.08:  # Tip is significantly below base
            return "down"
        else:
            return "sideways"
    
    def detect_thumbs_up(self, hand_landmarks) -> bool:
        """Detect thumbs up gesture: thumb up, other fingers curled."""
        # Check thumb direction
        if self.get_thumb_direction(hand_landmarks) != "up":
            return False
        
        # Check that other fingers are curled
        fingers_curled = all([
            self.is_finger_curled(hand_landmarks, HandLandmark.INDEX_TIP, HandLandmark.INDEX_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.MIDDLE_TIP, HandLandmark.MIDDLE_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.RING_TIP, HandLandmark.RING_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.PINKY_TIP, HandLandmark.PINKY_MCP),
        ])
        
        return fingers_curled
    
    def detect_thumbs_down(self, hand_landmarks) -> bool:
        """Detect thumbs down gesture: thumb down, other fingers curled."""
        # Check thumb direction
        if self.get_thumb_direction(hand_landmarks) != "down":
            return False
        
        # Check that other fingers are curled
        fingers_curled = all([
            self.is_finger_curled(hand_landmarks, HandLandmark.INDEX_TIP, HandLandmark.INDEX_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.MIDDLE_TIP, HandLandmark.MIDDLE_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.RING_TIP, HandLandmark.RING_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.PINKY_TIP, HandLandmark.PINKY_MCP),
        ])
        
        return fingers_curled
    
    def detect_raised_hand(self, hand_landmarks) -> bool:
        """Detect raised hand / open palm: all fingers extended."""
        fingers_extended = all([
            self.is_finger_extended(hand_landmarks, HandLandmark.THUMB_TIP, 
                                   HandLandmark.THUMB_IP, HandLandmark.THUMB_MCP, is_thumb=True),
            self.is_finger_extended(hand_landmarks, HandLandmark.INDEX_TIP,
                                   HandLandmark.INDEX_PIP, HandLandmark.INDEX_MCP),
            self.is_finger_extended(hand_landmarks, HandLandmark.MIDDLE_TIP,
                                   HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_MCP),
            self.is_finger_extended(hand_landmarks, HandLandmark.RING_TIP,
                                   HandLandmark.RING_PIP, HandLandmark.RING_MCP),
            self.is_finger_extended(hand_landmarks, HandLandmark.PINKY_TIP,
                                   HandLandmark.PINKY_PIP, HandLandmark.PINKY_MCP),
        ])
        
        return fingers_extended
    
    def get_hand_center(self, hand_landmarks) -> Tuple[float, float]:
        """Get the center point of the palm."""
        wrist = self.get_landmark_coords(hand_landmarks, HandLandmark.WRIST)
        middle_mcp = self.get_landmark_coords(hand_landmarks, HandLandmark.MIDDLE_MCP)
        
        center_x = (wrist[0] + middle_mcp[0]) / 2
        center_y = (wrist[1] + middle_mcp[1]) / 2
        
        return center_x, center_y
    
    def detect_clapping(self, hands_landmarks: List) -> bool:
        """Detect clapping: two hands close together."""
        if len(hands_landmarks) < 2:
            return False
        
        # Get centers of both hands
        center1 = self.get_hand_center(hands_landmarks[0])
        center2 = self.get_hand_center(hands_landmarks[1])
        
        # Calculate distance between hand centers
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Clapping detected when hands are close
        is_close = distance < 0.15
        
        # Track motion for more robust detection
        if self.prev_hand_distance is not None:
            # Hands coming together
            coming_together = self.prev_hand_distance > distance and distance < 0.2
            if coming_together and is_close:
                self.clap_cooldown = 10  # Keep showing clap for a few frames
        
        self.prev_hand_distance = distance
        
        if self.clap_cooldown > 0:
            self.clap_cooldown -= 1
            return True
            
        return is_close
    
    # ==================== Phase 3: Extended Gestures ====================
    
    def count_extended_fingers(self, hand_landmarks) -> int:
        """Count the number of extended fingers (0-5)."""
        count = 0
        
        # Thumb: check if extended (using existing method)
        if self.is_finger_extended(hand_landmarks, HandLandmark.THUMB_TIP,
                                   HandLandmark.THUMB_IP, HandLandmark.THUMB_MCP, is_thumb=True):
            count += 1
        
        # Index finger
        if self.is_finger_extended(hand_landmarks, HandLandmark.INDEX_TIP,
                                   HandLandmark.INDEX_PIP, HandLandmark.INDEX_MCP):
            count += 1
        
        # Middle finger
        if self.is_finger_extended(hand_landmarks, HandLandmark.MIDDLE_TIP,
                                   HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_MCP):
            count += 1
        
        # Ring finger
        if self.is_finger_extended(hand_landmarks, HandLandmark.RING_TIP,
                                   HandLandmark.RING_PIP, HandLandmark.RING_MCP):
            count += 1
        
        # Pinky finger
        if self.is_finger_extended(hand_landmarks, HandLandmark.PINKY_TIP,
                                   HandLandmark.PINKY_PIP, HandLandmark.PINKY_MCP):
            count += 1
        
        return count
    
    def detect_peace(self, hand_landmarks) -> bool:
        """Detect peace sign: index + middle extended, others curled."""
        # Index and middle must be extended
        index_extended = self.is_finger_extended(hand_landmarks, HandLandmark.INDEX_TIP,
                                                  HandLandmark.INDEX_PIP, HandLandmark.INDEX_MCP)
        middle_extended = self.is_finger_extended(hand_landmarks, HandLandmark.MIDDLE_TIP,
                                                   HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_MCP)
        
        # Ring and pinky must be curled
        ring_curled = self.is_finger_curled(hand_landmarks, HandLandmark.RING_TIP, HandLandmark.RING_MCP)
        pinky_curled = self.is_finger_curled(hand_landmarks, HandLandmark.PINKY_TIP, HandLandmark.PINKY_MCP)
        
        # Thumb can be in any position for peace sign
        return index_extended and middle_extended and ring_curled and pinky_curled
    
    def detect_ok_sign(self, hand_landmarks) -> bool:
        """Detect OK sign: thumb tip touches index tip, other fingers extended."""
        thumb_tip = self.get_landmark_coords(hand_landmarks, HandLandmark.THUMB_TIP)
        index_tip = self.get_landmark_coords(hand_landmarks, HandLandmark.INDEX_TIP)
        
        # Calculate distance between thumb tip and index tip
        distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                          (thumb_tip[1] - index_tip[1])**2)
        
        # Thumb and index should be close (forming circle)
        tips_close = distance < 0.08
        
        # Other fingers should be extended
        middle_extended = self.is_finger_extended(hand_landmarks, HandLandmark.MIDDLE_TIP,
                                                   HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_MCP)
        ring_extended = self.is_finger_extended(hand_landmarks, HandLandmark.RING_TIP,
                                                 HandLandmark.RING_PIP, HandLandmark.RING_MCP)
        pinky_extended = self.is_finger_extended(hand_landmarks, HandLandmark.PINKY_TIP,
                                                  HandLandmark.PINKY_PIP, HandLandmark.PINKY_MCP)
        
        return tips_close and middle_extended and ring_extended and pinky_extended
    
    def detect_pointing(self, hand_landmarks) -> bool:
        """Detect pointing: only index finger extended."""
        # Index must be extended
        index_extended = self.is_finger_extended(hand_landmarks, HandLandmark.INDEX_TIP,
                                                  HandLandmark.INDEX_PIP, HandLandmark.INDEX_MCP)
        
        # All other fingers must be curled
        middle_curled = self.is_finger_curled(hand_landmarks, HandLandmark.MIDDLE_TIP, HandLandmark.MIDDLE_MCP)
        ring_curled = self.is_finger_curled(hand_landmarks, HandLandmark.RING_TIP, HandLandmark.RING_MCP)
        pinky_curled = self.is_finger_curled(hand_landmarks, HandLandmark.PINKY_TIP, HandLandmark.PINKY_MCP)
        
        # Thumb direction shouldn't be up or down (to avoid confusion with thumbs up/down)
        thumb_dir = self.get_thumb_direction(hand_landmarks)
        thumb_sideways = thumb_dir == "sideways"
        
        return index_extended and middle_curled and ring_curled and pinky_curled and thumb_sideways
    
    def detect_fist(self, hand_landmarks) -> bool:
        """Detect fist: all fingers curled including thumb."""
        # All fingers must be curled
        fingers_curled = all([
            self.is_finger_curled(hand_landmarks, HandLandmark.INDEX_TIP, HandLandmark.INDEX_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.MIDDLE_TIP, HandLandmark.MIDDLE_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.RING_TIP, HandLandmark.RING_MCP),
            self.is_finger_curled(hand_landmarks, HandLandmark.PINKY_TIP, HandLandmark.PINKY_MCP),
        ])
        
        # Thumb should be sideways (tucked), not pointing up or down
        thumb_dir = self.get_thumb_direction(hand_landmarks)
        thumb_tucked = thumb_dir == "sideways"
        
        return fingers_curled and thumb_tucked
    
    def detect_rock_on(self, hand_landmarks) -> bool:
        """Detect rock on: index + pinky extended, middle + ring curled."""
        # Index and pinky must be extended
        index_extended = self.is_finger_extended(hand_landmarks, HandLandmark.INDEX_TIP,
                                                  HandLandmark.INDEX_PIP, HandLandmark.INDEX_MCP)
        pinky_extended = self.is_finger_extended(hand_landmarks, HandLandmark.PINKY_TIP,
                                                  HandLandmark.PINKY_PIP, HandLandmark.PINKY_MCP)
        
        # Middle and ring must be curled
        middle_curled = self.is_finger_curled(hand_landmarks, HandLandmark.MIDDLE_TIP, HandLandmark.MIDDLE_MCP)
        ring_curled = self.is_finger_curled(hand_landmarks, HandLandmark.RING_TIP, HandLandmark.RING_MCP)
        
        return index_extended and pinky_extended and middle_curled and ring_curled
    
    def classify_gesture(self, results, enabled_gestures: Optional[dict] = None, 
                         use_ml: Optional[bool] = None) -> Tuple[str, float]:
        """
        Classify the current gesture from MediaPipe results.
        
        Args:
            results: MediaPipe hand detection results
            enabled_gestures: Dict mapping gesture names to bool (enabled/disabled).
                              If None, only core gestures are enabled.
            use_ml: Override ML mode setting. If None, uses self.use_ml
        
        Returns:
            Tuple of (gesture_name, confidence). Confidence is 1.0 for heuristics.
        """
        if not results.multi_hand_landmarks:
            return "none", 0.0
        
        # Determine whether to use ML
        ml_mode = use_ml if use_ml is not None else self.use_ml
        
        # Use ML classification if enabled and model loaded
        if ml_mode and self.ml_model is not None:
            hand = results.multi_hand_landmarks[0]
            gesture, confidence = self.classify_gesture_ml(hand)
            
            # Check confidence threshold
            if confidence < self.ml_confidence_threshold:
                # Fall back to heuristics if confidence is low
                return self.classify_gesture_heuristic(results, enabled_gestures)
            
            # Check if gesture is enabled
            if enabled_gestures and not enabled_gestures.get(gesture, True):
                return "none", 0.0
            
            return gesture, confidence
        
        # Use heuristic classification
        return self.classify_gesture_heuristic(results, enabled_gestures)
    
    def classify_gesture_heuristic(self, results, enabled_gestures: Optional[dict] = None) -> Tuple[str, float]:
        """
        Classify gesture using heuristic rules.
        
        Returns:
            Tuple of (gesture_name, confidence). Confidence is always 1.0 for heuristics.
        """
        if not results.multi_hand_landmarks:
            return "none", 0.0
        
        # Default: core gestures enabled, extended gestures disabled
        if enabled_gestures is None:
            enabled_gestures = {g: not info.extended for g, info in GESTURES.items()}
        
        def is_enabled(gesture_name: str) -> bool:
            return enabled_gestures.get(gesture_name, True)
        
        hands = results.multi_hand_landmarks
        
        # Check for clapping first (requires two hands)
        if is_enabled("clapping") and self.detect_clapping(hands):
            return "clapping", 1.0
        
        # Check single-hand gestures on the first detected hand
        hand = hands[0]
        
        # === Core Gestures (Phase 1) ===
        if is_enabled("thumbs_up") and self.detect_thumbs_up(hand):
            return "thumbs_up", 1.0
        
        if is_enabled("thumbs_down") and self.detect_thumbs_down(hand):
            return "thumbs_down", 1.0
        
        if is_enabled("raised_hand") and self.detect_raised_hand(hand):
            return "raised_hand", 1.0
        
        # === Extended Gestures (Phase 3) ===
        # Check more specific gestures before generic ones
        
        # Rock on (index + pinky) - check before pointing
        if is_enabled("rock_on") and self.detect_rock_on(hand):
            return "rock_on", 1.0
        
        # Peace sign (index + middle) - check before finger count
        if is_enabled("peace") and self.detect_peace(hand):
            return "peace", 1.0
        
        # OK sign (thumb-index circle)
        if is_enabled("ok_sign") and self.detect_ok_sign(hand):
            return "ok_sign", 1.0
        
        # Pointing (only index)
        if is_enabled("pointing") and self.detect_pointing(hand):
            return "pointing", 1.0
        
        # Fist (all curled)
        if is_enabled("fist") and self.detect_fist(hand):
            return "fist", 1.0
        
        # Finger counting (check if any finger count gesture is enabled)
        finger_count_enabled = any(is_enabled(f"finger_count_{i}") for i in range(1, 6))
        if finger_count_enabled:
            count = self.count_extended_fingers(hand)
            if 1 <= count <= 5:
                gesture_name = f"finger_count_{count}"
                if is_enabled(gesture_name):
                    return gesture_name, 1.0
        
        return "none", 0.0
    
    def get_smoothed_gesture(self, raw_gesture: str) -> str:
        """Apply temporal smoothing using majority voting."""
        self.gesture_history.append(raw_gesture)
        
        if len(self.gesture_history) < 3:
            return raw_gesture
        
        # Count occurrences
        counts = {}
        for g in self.gesture_history:
            counts[g] = counts.get(g, 0) + 1
        
        # Return most common gesture
        return max(counts, key=counts.get)
    
    def process_frame(self, frame: np.ndarray, enabled_gestures: Optional[dict] = None, 
                      use_ml: Optional[bool] = None) -> Tuple[str, float, any]:
        """
        Process a frame and return the detected gesture.
        
        Args:
            frame: BGR image from OpenCV
            enabled_gestures: Dict mapping gesture names to bool (enabled/disabled)
            use_ml: Override ML mode setting
        
        Returns:
            Tuple of (smoothed_gesture_name, confidence, mediapipe_results)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Classify gesture (returns tuple of gesture, confidence)
        raw_gesture, confidence = self.classify_gesture(results, enabled_gestures, use_ml)
        smoothed_gesture = self.get_smoothed_gesture(raw_gesture)
        
        self.current_gesture = smoothed_gesture
        self.current_confidence = confidence
        
        return smoothed_gesture, confidence, results
    
    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks on frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
    
    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()


class ReactionOverlay:
    """Handles emoji reaction overlays on the video feed."""
    
    def __init__(self):
        self.active_reactions = []  # List of (emoji, x, y, start_time, duration)
        self.reaction_log = deque(maxlen=5)  # Recent reactions
        
    def trigger_reaction(self, gesture: str, frame_width: int, frame_height: int):
        """Add a new reaction to display."""
        if gesture == "none":
            return
            
        gesture_info = GESTURES.get(gesture)
        if not gesture_info or not gesture_info.emoji:
            return
        
        # Random position on right side of frame
        x = int(frame_width * 0.85)
        y = int(frame_height * 0.7)
        
        # Add to active reactions
        self.active_reactions.append({
            "emoji": gesture_info.emoji,
            "x": x,
            "y": y,
            "start_time": time.time(),
            "duration": 2.0,
        })
        
        # Log the reaction
        timestamp = time.strftime("%H:%M:%S")
        self.reaction_log.append(f"{gesture_info.emoji} {timestamp}")
    
    def draw_overlay(self, frame: np.ndarray, current_gesture: str) -> np.ndarray:
        """Draw reaction overlay on frame."""
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Update and draw active reactions (floating up animation)
        still_active = []
        for reaction in self.active_reactions:
            elapsed = current_time - reaction["start_time"]
            if elapsed < reaction["duration"]:
                # Calculate position (float upward)
                progress = elapsed / reaction["duration"]
                y_offset = int(progress * 100)
                alpha = 1.0 - progress  # Fade out
                
                # Draw emoji (using text as fallback since OpenCV doesn't render emoji natively)
                y_pos = reaction["y"] - y_offset
                
                # Draw a colored circle as emoji placeholder
                gesture_key = None
                for key, g in GESTURES.items():
                    if g.emoji == reaction["emoji"]:
                        gesture_key = key
                        break
                
                if gesture_key:
                    color = GESTURES[gesture_key].color
                    # Draw semi-transparent circle
                    overlay = frame.copy()
                    cv2.circle(overlay, (reaction["x"], y_pos), 30, color, -1)
                    cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
                    
                    # Draw emoji using PIL for proper rendering
                    frame = put_text_with_emoji(frame, reaction["emoji"], 
                                                (reaction["x"] - 20, y_pos - 25),
                                                font_size=48, color=(255, 255, 255))
                
                still_active.append(reaction)
        
        self.active_reactions = still_active
        
        # Draw current gesture status
        gesture_info = GESTURES.get(current_gesture, GESTURES["none"])
        status_text = f"Gesture: {gesture_info.name}"
        
        # Status background
        cv2.rectangle(frame, (10, 10), (350, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 60), gesture_info.color, 2)
        
        # Status text (no emoji - just clean text)
        cv2.putText(frame, status_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw reaction log
        log_y = 100
        cv2.putText(frame, "Recent:", (10, log_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for i, log_entry in enumerate(reversed(list(self.reaction_log))):
            frame = put_text_with_emoji(frame, log_entry, (10, log_y + 10 + i * 25),
                                        font_size=18, color=(180, 180, 180))
        
        # Draw controls help
        cv2.putText(frame, "Press 'q' to quit | 'd' for debug mode", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame


def main():
    """Main function to run the gesture recognition demo."""
    print("=" * 60)
    print("🖐️  Hand Gesture Recognition Demo")
    print("=" * 60)
    print("\nSupported Gestures:")
    print("  👍 Thumbs Up    - Thumb up, fingers curled")
    print("  👎 Thumbs Down  - Thumb down, fingers curled")
    print("  ✋ Raised Hand   - All fingers extended")
    print("  👏 Clapping     - Two hands close together")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'd' - Toggle debug mode (show landmarks)")
    print("=" * 60)
    
    # Initialize
    recognizer = GestureRecognizer(smoothing_window=7)
    overlay = ReactionOverlay()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    debug_mode = False
    last_gesture = "none"
    last_reaction_time = 0
    reaction_cooldown = 1.5  # Seconds between reactions
    
    print("\n✅ Webcam opened. Starting gesture recognition...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            gesture, results = recognizer.process_frame(frame)
            
            # Draw landmarks in debug mode
            if debug_mode:
                frame = recognizer.draw_landmarks(frame, results)
            
            # Trigger reaction on gesture change (with cooldown)
            current_time = time.time()
            if gesture != "none" and gesture != last_gesture:
                if current_time - last_reaction_time > reaction_cooldown:
                    overlay.trigger_reaction(gesture, frame.shape[1], frame.shape[0])
                    last_reaction_time = current_time
                    print(f"  → Detected: {GESTURES[gesture].emoji} {GESTURES[gesture].name}")
            
            last_gesture = gesture
            
            # Draw overlay
            frame = overlay.draw_overlay(frame, gesture)
            
            # Show debug indicator
            if debug_mode:
                cv2.putText(frame, "DEBUG MODE", (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Hand Gesture Recognition", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"  Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        print("✅ Resources released. Goodbye!")


if __name__ == "__main__":
    main()

