# 🖐️ **Project Plan: Real-Time Hand Gesture Recognition for Video Call Reactions**

### *(Hybrid Approach: Heuristics → Streamlit UI → Extended Gestures → ML)*

---

## **1. Project Summary**

This project builds a **real-time gesture recognition system** that detects common hand signals—**thumbs up, thumbs down, raised hand, and clapping**—using webcam video. It overlays corresponding **Zoom-style reactions** (👍 👎 👏 ✋) on the live feed.

The system uses **MediaPipe Hands** for 21-point hand landmark detection with a **five-phase approach**:

1. **Phase 1 (Complete):** Heuristic-based detection for 4 core gestures using geometric rules
2. **Phase 2 (Planned):** Streamlit UI for interactive web-based demo
3. **Phase 3 (Planned):** Extended gesture vocabulary with heuristics (finger counting, ✌️ 👌 👆 👋 ✊)
4. **Phase 4 (Planned):** Data collection tool for building ML training dataset
5. **Phase 5 (Planned):** ML classifier training and integration

The project demonstrates:

* Real-time CV processing
* Rule-based gesture recognition using geometric heuristics
* Interactive web deployment (Streamlit)
* Dataset collection & labeling
* Model training + evaluation

---

## **2. Supported Gestures**

### **Core Gestures (Phase 1) ✅**

| Gesture     | Emoji | Description                                  |
| ----------- | ----- | -------------------------------------------- |
| Thumbs Up   | 👍    | Thumb extended upward, other fingers curled  |
| Thumbs Down | 👎    | Thumb extended downward, others curled       |
| Raised Hand | ✋    | All fingers extended, palm upright           |
| Clapping    | 👏    | Two hands close together                     |

### **Extended Gestures (Phase 3) — Toggleable**

| Gesture       | Emoji   | Description                              | Default |
| ------------- | ------- | ---------------------------------------- | ------- |
| Finger Count  | 1️⃣-5️⃣   | Count of extended fingers                | OFF     |
| Peace Sign    | ✌️      | Index + middle extended, others curled   | OFF     |
| OK Sign       | 👌      | Thumb-index circle, others extended      | OFF     |
| Pointing      | 👆      | Only index finger extended               | OFF     |
| Fist          | ✊      | All fingers curled, thumb tucked         | OFF     |
| Rock On       | 🤘      | Index + pinky extended, others curled    | OFF     |

---

## **3. Phase 1: Heuristic-Based Detection (Complete)**

The initial implementation uses **geometric heuristics** on MediaPipe landmarks to detect the 4 core gestures. This approach works well because these gestures are geometrically distinct.

### **Detection Rules**

| Gesture | Heuristic Logic |
|---------|-----------------|
| 👍 Thumbs Up | Thumb tip significantly above thumb MCP + all other fingers curled (tips below MCPs) |
| 👎 Thumbs Down | Thumb tip significantly below thumb MCP + all other fingers curled |
| ✋ Raised Hand | All 5 fingers extended (tips above PIPs) |
| 👏 Clapping | Two hands detected + palm centers within threshold distance |

### **Why Heuristics Work for Core Gestures**

* **Geometrically distinct poses** — Each gesture has a unique spatial configuration
* **Low ambiguity** — Thumbs up/down differ only in vertical thumb orientation
* **Robust landmarks** — MediaPipe provides reliable 21-point tracking
* **Zero training data** — Works immediately without data collection
* **Fast inference** — Simple coordinate comparisons, no model overhead

### **Current Pipeline**

1. Capture webcam frame
2. Detect hands → extract 21 landmarks per hand
3. Apply geometric rules to classify gesture
4. Temporal smoothing (majority vote over 7 frames)
5. Gesture → emoji mapping
6. Display overlay

---

## **4. Phase 2: Streamlit UI (Planned)**

Build an interactive web-based interface using Streamlit for a polished demo experience.

### **UI Features**

| Feature | Description |
|---------|-------------|
| **Live webcam feed** | Real-time video with gesture overlay |
| **Gesture toggle panel** | Enable/disable individual gestures |
| **Reaction history** | Scrollable log of detected gestures with timestamps |
| **Debug mode toggle** | Show/hide hand landmarks |
| **Statistics dashboard** | Gesture counts, detection rate, FPS |
| **Settings panel** | Adjust detection sensitivity, smoothing window |

### **Why Streamlit?**

* **Rapid prototyping** — Build interactive UIs in pure Python
* **Built-in webcam support** — `st.camera_input()` for easy video capture
* **Clean aesthetics** — Modern, professional look out of the box
* **Easy deployment** — One-click deploy to Streamlit Cloud
* **Portfolio-ready** — Shareable link for applications/interviews

### **Technical Approach**

```
streamlit run app.py
```

* Use `st.sidebar` for controls and settings
* `st.empty()` containers for real-time video updates
* Session state for tracking gesture history and stats

---

## **5. Phase 3: Extended Gesture Vocabulary (Planned)**

Expand the gesture library using heuristic rules, with **toggle controls** to enable/disable each gesture.

### **New Gestures (Heuristic-Based)**

| Gesture | Emoji | Heuristic Logic | Toggle Default |
|---------|-------|-----------------|----------------|
| **Finger Count (1-5)** | 1️⃣-5️⃣ | Count extended fingers | OFF |
| **Peace Sign** | ✌️ | Index + middle extended, others curled | OFF |
| **OK Sign** | 👌 | Thumb tip close to index tip, other fingers extended | OFF |
| **Pointing** | 👆 | Only index extended | OFF |
| **Fist** | ✊ | All fingers curled, thumb tucked | OFF |
| **Rock On** | 🤘 | Index + pinky extended, others curled | OFF |

### **Finger Counting Logic**

```python
def count_extended_fingers(hand_landmarks):
    """Count how many fingers are extended (0-5)."""
    count = 0
    
    # Thumb: check horizontal distance from palm
    if is_thumb_extended(hand_landmarks):
        count += 1
    
    # Fingers: tip above PIP joint = extended
    for finger in [INDEX, MIDDLE, RING, PINKY]:
        if finger_tip.y < finger_pip.y:
            count += 1
    
    return count
```

### **Toggle System**

Users can enable/disable gestures via the Streamlit UI:

* **Core gestures** (👍 👎 ✋ 👏) — Always ON by default
* **Extended gestures** (✌️ 👌 👆 ✊ 🤘) — OFF by default, toggle ON as needed
* **Finger counting** — OFF by default (can conflict with other gestures)

This prevents gesture conflicts and lets users customize for their use case.

### **Observed Limitations: Why ML Is Needed**

After implementing extended gestures with heuristics, we observed significant detection conflicts that validate the need for ML:

| Conflict | Gestures Affected | Root Cause |
|----------|-------------------|------------|
| **Extended fingers overlap** | ✌️ Peace vs ✋ Raised Hand | Both have index + middle extended; peace requires detecting ring/pinky curl precisely |
| **Single finger ambiguity** | 👆 Pointing vs 👍 Thumbs Up | Both involve one extended digit with others curled; thumb position is subtle |
| **Curled fingers similarity** | ✊ Fist vs base of thumbs gestures | All have curled fingers; only thumb orientation differs |
| **Counting conflicts** | 🔢 Finger count vs named gestures | 5 fingers = raised hand, 2 fingers = peace, 1 finger = pointing |
| **Orientation sensitivity** | All gestures | Heuristics assume upright hand; rotated hands break thresholds |

**Key Insight:** Heuristics excel at **geometrically distinct** poses (thumbs up vs open palm) but fail when gestures share structural similarities. The rules become increasingly brittle as more gestures are added.

**ML Advantages:**
* Learns subtle differences from real examples rather than hand-coded thresholds
* Handles natural variation in how different people perform gestures
* Provides confidence scores to handle ambiguous cases gracefully
* Generalizes across hand orientations and distances

This observation motivates **Phase 4 (Data Collection)** and **Phase 5 (ML Training)** to achieve production-quality recognition for the full gesture vocabulary.

---

## **6. Phase 4: Data Collection Tool (Planned)**

Build a dedicated tool to collect and manage video data for ML training.

### **Data Collection App Features**

| Feature | Description |
|---------|-------------|
| **Recording interface** | Live webcam with start/stop recording |
| **Gesture label selection** | Dropdown or hotkeys to tag current gesture |
| **Session management** | Name sessions, track participants |
| **Progress tracker** | Show samples collected per gesture class |
| **Data preview** | Review recorded clips before saving |
| **Export options** | Save as CSV (landmarks) or video clips |

### **Data Organization**

```
data/
├── sessions/
│   ├── session_001_alice/
│   │   ├── metadata.json       # participant info, date, settings
│   │   ├── thumbs_up/
│   │   │   ├── clip_001.csv    # landmark coordinates per frame
│   │   │   ├── clip_002.csv
│   │   ├── peace_sign/
│   │   └── ...
│   ├── session_002_bob/
│   └── ...
├── combined/
│   └── all_landmarks.csv       # merged dataset for training
└── stats.json                  # collection progress summary
```

### **Metadata Tracking**

```json
{
  "session_id": "session_001",
  "participant": "Alice",
  "date": "2024-01-15",
  "duration_minutes": 12,
  "samples_collected": {
    "thumbs_up": 156,
    "thumbs_down": 142,
    "peace_sign": 98,
    "...": "..."
  },
  "lighting": "natural",
  "distance": "medium",
  "hand": "right"
}
```

### **Collection Guidelines**

* **Target:** 100-300 samples per gesture per participant
* **Variation:** Multiple distances, lighting conditions, both hands
* **Quality:** Auto-skip frames where hand detection fails

---

## **7. Phase 5: ML Classifier (Planned)**

Train and integrate a machine learning classifier for robust gesture recognition.

### **Why Add ML After Heuristics?**

| Limitation of Heuristics | ML Solution |
|--------------------------|-------------|
| **Orientation sensitivity** — Rules assume upright hand | ML generalizes across orientations |
| **Inter-user variability** — Fixed thresholds don't fit everyone | ML adapts to population variation |
| **Edge cases** — Hard to write rules for ambiguous poses | ML learns from examples |
| **Confidence scores** — Heuristics are binary yes/no | ML provides probabilities |

### **Model Selection Decisions**

#### **Primary: Random Forest** ✅ Recommended

| Aspect | Details |
|--------|---------|
| **Why** | Fast training & inference, handles multiclass natively, no hyperparameter sensitivity |
| **Training** | O(n × m × log n) — trains in seconds |
| **Inference** | O(tree_depth × n_trees) — just tree traversals, very fast |
| **Multiclass** | Native support (no OvO/OvA strategy needed) |
| **Settings** | n_estimators=200, max_depth=20, class_weight='balanced' |

#### **Secondary: SVM with RBF Kernel**

| Aspect | Details |
|--------|---------|
| **Why** | Strong performance on small-medium datasets, good generalization |
| **Multiclass Strategy** | **One-vs-One (OvO)** — scikit-learn default |
| **OvO Explained** | Trains k×(k-1)/2 binary classifiers (45 for 10 gesture classes) |
| **Why OvO over OvA?** | Each OvO classifier sees balanced binary data; OvA creates imbalanced problems (1 class vs ALL others) |
| **Settings** | kernel='rbf', C=1.0, probability=True |

#### **Why NOT Neural Networks?**

| Reason | Explanation |
|--------|-------------|
| **Dataset size** | Landmark data is small (1000s of samples) — DNNs need more |
| **Feature space** | Only 63 features (21 landmarks × 3 coords) — not high-dimensional |
| **Interpretability** | RF/SVM easier to debug than black-box neural nets |
| **Deployment** | No GPU, PyTorch/TensorFlow dependencies needed |
| **Speed** | RF inference is microseconds; neural nets add latency |

### **Feature Space**

* **Input:** 63 normalized features (x, y, z for 21 landmarks)
* **Normalization:** Wrist-centered, unit-scaled (done during data collection)
* **No additional feature engineering** — landmarks are already good features

### **Pipeline (with ML)**

1. Capture webcam frame
2. Detect hands → extract 21 landmarks
3. Normalize landmarks (wrist = origin, scale to unit)
4. ML classifier predicts gesture + confidence
5. Temporal smoothing + confidence thresholding
6. Display overlay

### **Integration Strategy**

* **Hybrid mode:** Use heuristics as fallback when ML confidence is low
* **A/B comparison:** Toggle between heuristic and ML modes in UI
* **Gradual rollout:** Start with ML for extended gestures only

**Backend:** Python, OpenCV, MediaPipe, scikit-learn

**Frontend:** Streamlit

---

## **8. ML Data Collection Details** *(Phase 4-5 Reference)*

Gesture classifiers built on MediaPipe landmarks need **far less data** than image-based models because the feature space is low-dimensional and highly structured.

Below is a full professional-grade data collection workflow.

---

### **8.1. Data Requirements**

To get a clean demo-quality classifier:

#### **Minimum Viable (For Demo Only)**

* **50–100 samples per gesture**
* Total ≈ **250–500 samples**

#### **Robust (Recommended for Application Portfolio)**

* **300–500 samples per gesture**
* Across **3–5 different people**
* Total: **1500–2500 samples**

This size is easy to collect:

* 10 seconds per gesture per person
* At ~15 FPS → 150 frames per gesture per person
* 3–4 volunteers → 450–600 frames per gesture

---

### **8.2. Data Collection Script**

Write a Python tool that:

* Displays webcam feed with MediaPipe landmarks
* Lets user select label using keyboard keys
* Saves each frame's **normalized landmark coordinates** to a CSV

#### **Example UI Mapping**

| Key | Label       |
| --- | ----------- |
| `1` | thumbs_up   |
| `2` | thumbs_down |
| `3` | raised_hand |
| `4` | clap        |
| `0` | none        |

#### **Data Saved Per Frame**

```
x1, y1, x2, y2, …, x21, y21, label
```

If including depth info:

```
x1, y1, z1, x2, y2, z2, …, label
```

#### **Normalization Strategy**

Normalize per hand:

* Translate so wrist = origin
* Scale so max distance from wrist = 1
* Optional: rotate to align palm orientation

This makes training more robust across users and distances from camera.

---

### **8.3. How to Capture High-Quality Data**

#### **A. Collect "steady pose" samples**

The participant holds the gesture for 5–10 seconds.

Avoid:

* mid-transition frames
* frames where gesture is unclear
* occlusion

#### **B. Use multiple distances**

Ask participants to record at:

* Close-up (face distance)
* Mid-distance (upper torso)
* Far distance (full upper body)

This helps generalization.

#### **C. Collect left and right hand data**

Gesture detection should work ambidextrously.

#### **D. Vary background & lighting**

Record in:

* bright light
* dimmer light
* cluttered vs plain backgrounds

Even though landmarks abstract away pixels, this still helps reduce tracking failures.

#### **E. For clapping**

Collect **motion sequences**, not static poses:

* Record 5–10 seconds of clapping
* Keep all frames
* Later compute temporal differences if needed (e.g., distance between palms decreasing/increasing)

---

### **8.4. Cleaning & Preparing the Dataset**

After recording, process each CSV:

1. **Drop frames where hand detection failed**
   (Some rows will have NaNs if MediaPipe didn't detect the hand.)

2. **Remove duplicates and static frames**
   For clapping, keep frames where movement exists.

3. **Ensure even class distribution**
   If gestures have 500 samples but "none" has 100, pad "none" using extra recordings.

4. **Optionally augment the landmarks**
   * Add Gaussian noise (tiny jitter)
   * Random small rotations
   * Mirror left-hand ↔ right-hand coordinates

This improves model robustness.

---

### **8.5. Model Training**

#### **Models that work well**

* **SVM (RBF kernel)**
* **RandomForest (n_estimators=200–300)**
* **Tiny MLP**:
  * 2–3 dense layers of 64–128 units

#### **Training Workflow**

1. Load all CSVs
2. Shuffle
3. Train/validation split: **80/20**
4. Train model
5. Evaluate on validation set:
   * Accuracy
   * Per-class recall
   * Confusion matrix

#### **Typical expected performance**

With ~300 samples/gesture:

* **Accuracy:** 90–98%
* **Clapping** may be lower due to motion, but network still detects the presence of two hands very well.

---

### **8.6. Saving the Model**

Use joblib:

```python
import joblib
joblib.dump(model, "gesture_classifier.pkl")
```

At runtime:

```python
model = joblib.load("gesture_classifier.pkl")
prediction = model.predict([landmark_vector])
```

---

## **9. Real-Time System**

### **Inference Pipeline**

**Phase 1-3 (Heuristics):**
1. Webcam frame → MediaPipe → hand landmarks
2. Apply geometric rules for gesture classification
3. Apply **temporal smoothing** (majority vote over last 7 frames)
4. Display reaction overlay

**Phase 5 (ML):**
1. Webcam frame → MediaPipe → hand landmarks
2. Normalize landmarks → predict with classifier
3. Apply temporal smoothing + confidence thresholding
4. Display reaction overlay

### **Overlays**

* Emoji floats up & fades
* Reaction log in sidebar ("👍 detected at 12:36:22")
* Hand skeleton rendering for debugging (toggle with 'd' key)

---

## **10. Timeline**

### **Phase 1: Core Heuristics (Complete) — 3 Days**

| Day | Tasks |
|-----|-------|
| 1 | Webcam + MediaPipe setup, landmark extraction, visualization |
| 2 | Implement geometric heuristics for 4 gestures, temporal smoothing |
| 3 | Add emoji overlay with PIL rendering, debug mode |

**✅ Deliverable:** Working OpenCV demo with 4 gesture recognition

---

### **Phase 2: Streamlit UI — 2 Days**

| Day | Tasks |
|-----|-------|
| 4 | Build Streamlit app structure, webcam integration, basic layout |
| 5 | Add gesture toggle panel, reaction history, settings, polish UI |

**Deliverable:** Interactive web-based demo

---

### **Phase 3: Extended Gestures — 2 Days**

| Day | Tasks |
|-----|-------|
| 6 | Implement finger counting + new gesture heuristics (✌️ 👌 👆 ✊ 🤘) |
| 7 | Add gesture toggles to UI, handle conflicts, test all gestures |

**Deliverable:** Expanded gesture library with toggle controls

---

### **Phase 4: Data Collection Tool — 2 Days**

| Day | Tasks |
|-----|-------|
| 8 | Build data collection interface with recording, labeling, session management |
| 9 | Add progress tracking, data preview, export functionality |

**Deliverable:** Tool to collect and organize training data

---

### **Phase 5: ML Integration — 3 Days**

| Day | Tasks |
|-----|-------|
| 10 | Collect initial dataset (~100 samples per gesture) |
| 11 | Train classifier (SVM/RandomForest), evaluate, iterate |
| 12 | Integrate ML model, add heuristic fallback, final testing |

**Deliverable:** Hybrid heuristic + ML gesture recognition system

---

## **11. Extensions (Optional)**

### **Comparison & Evaluation**
* **Live comparison mode** — Side-by-side heuristic vs ML predictions with agreement tracking
* **Accuracy benchmarking** — Automated comparison on held-out test set
* **Confusion matrix visualization** — Interactive heatmap of misclassifications

### **Additional Gestures**
* **Wave detection** — Side-to-side motion with temporal analysis (LSTM/1D CNN)
* **Custom gesture training** — Let users define and train their own gestures
* **Two-hand gestures** — Heart shape, timeout signal, etc.

### **Audio & Multimodal**
* **Audio-assisted clap detection** — Combine visual + audio for robust clapping
* **Voice command integration** — "Hey, thumbs up!" triggers gesture mode

### **Deployment & Scale**
* **TensorFlow.js port** — Run entirely in browser, no Python backend
* **Mobile app** — React Native + MediaPipe for iOS/Android
* **Video conferencing plugin** — Zoom/Teams integration for live reactions

### **Advanced CV**
* **Multi-person gesture recognition** — Track multiple hands/people simultaneously
* **Full-body gestures** — Integrate with OpenPose or Detectron2
* **Depth camera support** — Intel RealSense for 3D hand tracking

---

## **12. How to Present This in a Master's Application**

> "I built a real-time gesture recognition system for video-call reactions using MediaPipe hand landmarks. I took a principled engineering approach: starting with heuristic-based detection using geometric rules to recognize core gestures with zero training data, then building an interactive Streamlit UI, expanding to 10+ gestures including finger counting, and finally training a machine learning classifier on custom-collected data for improved robustness. I also built a data collection tool to systematically gather and organize training samples. The project demonstrates my ability to choose the right tool for each problem—simple rules when they suffice, ML when needed—and to build complete end-to-end systems."

This reads extremely strong to admissions committees — it shows engineering judgment, full-stack skills, and the ability to iterate on solutions.

---

## **13. Project Status**

### **Phase 1: Core Heuristics — Complete ✅**

- ✅ Webcam + MediaPipe integration
- ✅ Heuristic gesture detection (4 gestures: 👍 👎 ✋ 👏)
- ✅ Temporal smoothing (majority vote)
- ✅ Emoji overlay with PIL rendering
- ✅ Debug mode with landmark visualization

### **Phase 2: Streamlit UI — Complete ✅**

- ✅ Streamlit app structure
- ✅ Live webcam feed in browser
- ✅ Gesture toggle panel
- ✅ Reaction history log
- ✅ Settings panel (detection confidence, smoothing, cooldown)
- ✅ Statistics dashboard (FPS, frame count, gesture counts)

### **Phase 3: Extended Gestures — Complete ✅**

- ✅ Finger counting (1️⃣-5️⃣)
- ✅ Peace sign (✌️)
- ✅ OK sign (👌)
- ✅ Pointing (👆)
- ✅ Fist (✊)
- ✅ Rock on (🤘)
- ✅ Gesture toggle controls in UI (extended gestures OFF by default)

### **Phase 4: Data Collection Tool — Complete ✅**

- ✅ Recording interface (live webcam with landmark visualization)
- ✅ Label selection system (button per gesture class)
- ✅ Session management (create/load sessions with metadata)
- ✅ Progress tracking (samples per gesture with targets)
- ✅ Data export (combined CSV with normalized landmarks)

### **Phase 5: ML Classifier — Complete ✅**

- ✅ Training UI (`train_model.py`) with Random Forest & SVM
- ✅ Model comparison and evaluation metrics
- ✅ Model export and "Set as Active" workflow
- ✅ Integrated ML classifier into gesture recognition
- ✅ Hybrid mode: ML with heuristic fallback when confidence low
- ✅ ML toggle in Streamlit UI with confidence threshold control

