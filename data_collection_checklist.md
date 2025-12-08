# Data Collection Checklist

**Date:** ________________  
**Location:** ________________

---

## Setup (Before Starting)

### Per Laptop Setup
- [ ] Clone/copy the project to each laptop
- [ ] Conda environment activated: `conda activate hands`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Test camera works: `streamlit run data_collection.py`
- [ ] Create `data/` folder exists

### Naming Convention (CRITICAL - Prevents Overlap)

Each person uses this format for session names:

| Laptop | Person | Session Name Format |
|--------|--------|---------------------|
| Laptop 1 | Person A | `session_A_right`, `session_A_left` |
| Laptop 2 | Person B | `session_B_right`, `session_B_left` |
| Laptop 3 | Person C | `session_C_right`, `session_C_left` |

**Each person creates 2 sessions:**
1. `session_[INITIAL]_right` — All gestures with RIGHT hand
2. `session_[INITIAL]_left` — All gestures with LEFT hand

---

## Gestures to Collect

**1 clip = 3-5 seconds ≈ 100-150 frames (samples)**

### Core Gestures (Required)
| # | Gesture | Emoji | Clips per Hand | ≈ Samples |
|---|---------|-------|----------------|-----------|
| 1 | Thumbs Up | 👍 | **2 clips** | ~200-300 |
| 2 | Thumbs Down | 👎 | **2 clips** | ~200-300 |
| 3 | Raised Hand | ✋ | **2 clips** | ~200-300 |
| 4 | Clapping | 👏 | **2 clips** (both hands) | ~200-300 |

### Extended Gestures (Required)
| # | Gesture | Emoji | Clips per Hand | ≈ Samples |
|---|---------|-------|----------------|-----------|
| 5 | Peace | ✌️ | **1-2 clips** | ~100-200 |
| 6 | OK Sign | 👌 | **1-2 clips** | ~100-200 |
| 7 | Pointing | 👆 | **1-2 clips** | ~100-200 |
| 8 | Fist | ✊ | **1-2 clips** | ~100-200 |
| 9 | Rock On | 🤘 | **1-2 clips** | ~100-200 |

### Background (Required)
| # | Gesture | Clips | ≈ Samples |
|---|---------|-------|-----------|
| 10 | None (no hand / random poses) | **2 clips** | ~200-300 |

---

## Collection Protocol

### Recording Guidelines
- **Duration per clip:** 3-5 seconds (hold steady the whole time)
- **Clips per gesture:** 2 clips (vary distance/angle between clips)
- **Wait for "Hand Detected"** (green text) before starting
- **Hold steady** — don't move while recording
- **Vary between clips:** Clip 1 closer, Clip 2 farther (or different angle)

### Distance Variation
Between your 2 clips for each gesture, vary the distance:
- **Clip 1:** Medium distance (arm's length) 
- **Clip 2:** Close OR far (hand fills frame OR upper body visible)

### Lighting
- Face a window or lamp (front lighting)
- Avoid backlight (window behind you)
- All 3 laptops should have similar lighting if possible

---

## Per-Person Checklist

**Remember: 1 clip = 3-5 seconds of holding the gesture steady**

### Person A (Laptop 1)

**Session: `session_A_right` (RIGHT hand)**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👍 Thumbs Up | 2 | [ ] [ ] |
| 👎 Thumbs Down | 2 | [ ] [ ] |
| ✋ Raised Hand | 2 | [ ] [ ] |
| ✌️ Peace | 2 | [ ] [ ] |
| 👌 OK Sign | 2 | [ ] [ ] |
| 👆 Pointing | 2 | [ ] [ ] |
| ✊ Fist | 2 | [ ] [ ] |
| 🤘 Rock On | 2 | [ ] [ ] |
| ❌ None | 2 | [ ] [ ] |

**Session: `session_A_left` (LEFT hand)**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👍 Thumbs Up | 2 | [ ] [ ] |
| 👎 Thumbs Down | 2 | [ ] [ ] |
| ✋ Raised Hand | 2 | [ ] [ ] |
| ✌️ Peace | 2 | [ ] [ ] |
| 👌 OK Sign | 2 | [ ] [ ] |
| 👆 Pointing | 2 | [ ] [ ] |
| ✊ Fist | 2 | [ ] [ ] |
| 🤘 Rock On | 2 | [ ] [ ] |
| ❌ None | 2 | [ ] [ ] |

**👏 Clapping (both hands) — in `session_A_right`**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👏 Clapping | 2 | [ ] [ ] |

**Person A Total: 38 clips (~15-20 minutes)**

---

### Person B (Laptop 2)

**Session: `session_B_right` (RIGHT hand)**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👍 Thumbs Up | 2 | [ ] [ ] |
| 👎 Thumbs Down | 2 | [ ] [ ] |
| ✋ Raised Hand | 2 | [ ] [ ] |
| ✌️ Peace | 2 | [ ] [ ] |
| 👌 OK Sign | 2 | [ ] [ ] |
| 👆 Pointing | 2 | [ ] [ ] |
| ✊ Fist | 2 | [ ] [ ] |
| 🤘 Rock On | 2 | [ ] [ ] |
| ❌ None | 2 | [ ] [ ] |

**Session: `session_B_left` (LEFT hand)**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👍 Thumbs Up | 2 | [ ] [ ] |
| 👎 Thumbs Down | 2 | [ ] [ ] |
| ✋ Raised Hand | 2 | [ ] [ ] |
| ✌️ Peace | 2 | [ ] [ ] |
| 👌 OK Sign | 2 | [ ] [ ] |
| 👆 Pointing | 2 | [ ] [ ] |
| ✊ Fist | 2 | [ ] [ ] |
| 🤘 Rock On | 2 | [ ] [ ] |
| ❌ None | 2 | [ ] [ ] |

**👏 Clapping (both hands) — in `session_B_right`**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👏 Clapping | 2 | [ ] [ ] |

**Person B Total: 38 clips (~15-20 minutes)**

---

### Person C (Laptop 3)

**Session: `session_C_right` (RIGHT hand)**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👍 Thumbs Up | 2 | [ ] [ ] |
| 👎 Thumbs Down | 2 | [ ] [ ] |
| ✋ Raised Hand | 2 | [ ] [ ] |
| ✌️ Peace | 2 | [ ] [ ] |
| 👌 OK Sign | 2 | [ ] [ ] |
| 👆 Pointing | 2 | [ ] [ ] |
| ✊ Fist | 2 | [ ] [ ] |
| 🤘 Rock On | 2 | [ ] [ ] |
| ❌ None | 2 | [ ] [ ] |

**Session: `session_C_left` (LEFT hand)**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👍 Thumbs Up | 2 | [ ] [ ] |
| 👎 Thumbs Down | 2 | [ ] [ ] |
| ✋ Raised Hand | 2 | [ ] [ ] |
| ✌️ Peace | 2 | [ ] [ ] |
| 👌 OK Sign | 2 | [ ] [ ] |
| 👆 Pointing | 2 | [ ] [ ] |
| ✊ Fist | 2 | [ ] [ ] |
| 🤘 Rock On | 2 | [ ] [ ] |
| ❌ None | 2 | [ ] [ ] |

**👏 Clapping (both hands) — in `session_C_right`**
| Gesture | Clips | Done? |
|---------|-------|-------|
| 👏 Clapping | 2 | [ ] [ ] |

**Person C Total: 38 clips (~15-20 minutes)**

---

## After Collection

### On Each Laptop
1. [ ] Check Dashboard tab — verify all gestures have samples
2. [ ] Click "Export Combined CSV" (creates `data/combined/all_landmarks.csv`)
3. [ ] Copy entire `data/sessions/` folder to USB drive

### Aggregation (On Main Machine)
1. [ ] Create master `data/sessions/` folder
2. [ ] Copy all session folders from each laptop:
   ```
   data/sessions/
   ├── session_A_right/
   ├── session_A_left/
   ├── session_B_right/
   ├── session_B_left/
   ├── session_C_right/
   └── session_C_left/
   ```
3. [ ] Run data collection tool: `streamlit run data_collection.py`
4. [ ] Go to Dashboard tab — should show all 3 participants
5. [ ] Click "Export Combined CSV" — creates merged dataset
6. [ ] Verify sample counts in Dashboard

---

## Expected Totals

| Gesture | Clips/Person | × 3 People | × 2 Hands | Total Clips | ≈ Samples |
|---------|--------------|------------|-----------|-------------|-----------|
| Thumbs Up | 2 | 6 | 12 | **12** | ~1,500 |
| Thumbs Down | 2 | 6 | 12 | **12** | ~1,500 |
| Raised Hand | 2 | 6 | 12 | **12** | ~1,500 |
| Clapping | 2 | 6 | — | **6** | ~750 |
| Peace | 2 | 6 | 12 | **12** | ~1,500 |
| OK Sign | 2 | 6 | 12 | **12** | ~1,500 |
| Pointing | 2 | 6 | 12 | **12** | ~1,500 |
| Fist | 2 | 6 | 12 | **12** | ~1,500 |
| Rock On | 2 | 6 | 12 | **12** | ~1,500 |
| None | 2 | 6 | 12 | **12** | ~1,500 |

**Grand Total: 114 clips ≈ 14,000+ samples**

---

## Time Estimate

| Task | Time |
|------|------|
| Setup per laptop | 5 min |
| Right hand (18 clips × 5 sec) | 10 min |
| Left hand (18 clips × 5 sec) | 10 min |
| Clapping (2 clips) | 2 min |
| Buffer/switching gestures | 5 min |
| **Total per person** | **~30 min** |

With 3 people working in parallel: **~30-40 minutes total**

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No Hand Detected" | Improve lighting, move hand closer |
| Low FPS | Close other apps, reduce resolution |
| Session name conflict | Use unique initials (A, B, C) |
| Camera not found | Check permissions, restart app |
| Missing samples | Check Dashboard before exporting |

---

## Quick Reference Card (Print This!)

```
┌─────────────────────────────────────────────────────────┐
│  DATA COLLECTION QUICK REFERENCE                        │
├─────────────────────────────────────────────────────────┤
│  1. Create session: session_[YOUR_INITIAL]_[left/right] │
│  2. Click START CAMERA                                  │
│  3. Select gesture button (e.g., 👍 Thumbs Up)          │
│  4. Wait for "Hand Detected" (green text)               │
│  5. Click 🔴 START Recording                            │
│  6. Hold gesture steady for 3-5 seconds                 │
│  7. Click ⏹️ STOP Recording                             │
│  8. Repeat for second clip                              │
│  9. Move to next gesture                                │
├─────────────────────────────────────────────────────────┤
│  TARGET: 2 clips per gesture (each 3-5 seconds)         │
│  TOTAL: 38 clips per person (~20 minutes)               │
└─────────────────────────────────────────────────────────┘
```

---

*Good luck with data collection! 🖐️*

