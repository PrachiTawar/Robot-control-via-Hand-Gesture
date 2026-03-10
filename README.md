# 🤖 AI Gesture Controlled Robot Simulator

Control a virtual robot in real time using nothing but your hand gestures — powered by **MediaPipe**, **OpenCV**, and **Streamlit**.

---

## ✨ Overview

The system captures live video from your webcam, detects hand landmarks with Google's MediaPipe Hands model, classifies the current gesture, and maps it to a robot movement command displayed instantly in the UI.

| Gesture | Robot Command |
|---|---|
| 👈 Index finger pointing **LEFT** | Move **LEFT** |
| 👉 Index finger pointing **RIGHT** | Move **RIGHT** |
| 👍 **Thumb UP** (fist, thumb raised) | Move **FORWARD** |
| 👎 **Thumb DOWN** (fist, thumb lowered) | Move **BACKWARD** |
| ✋ **Open palm** (all fingers extended) | **STOP** |

---

## 📁 Project Structure

```
gesture_robot/
├── app.py               # Streamlit UI & webcam loop
├── gesture_control.py   # MediaPipe detection + gesture logic
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🛠️ Installation

### 1 · Clone or download the project

```bash
git clone https://github.com/your-username/gesture-robot.git
cd gesture-robot
```

Or simply place the four project files in a folder named `gesture_robot/`.

### 2 · Create a virtual environment (recommended)

```bash
python -m venv .venv

# Activate — macOS / Linux
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9 – 3.11** is recommended. MediaPipe does not yet officially support Python 3.12+ on all platforms.

---

## 🚀 How to Run

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).  
Open it in your browser, then:

1. Click **▶ Start Camera** in the left sidebar.
2. Allow browser/OS webcam access if prompted.
3. Hold your hand in front of the camera and try the gestures listed above.
4. The detected command appears in the right panel and as an overlay on the video feed.
5. Click **⏹ Stop Camera** when done.

---

## ⚙️ Configuration

| Sidebar control | Description |
|---|---|
| **Camera index** | `0` = default webcam. Set `1`, `2`, … for external cameras. |
| **Detection confidence** | MediaPipe minimum detection threshold (0.4 – 1.0). |

---

## 🧠 How It Works

### `gesture_control.py`

```
WebcamFrame
    │
    ▼
cv2.flip()          ← mirror so gestures feel natural
    │
    ▼
MediaPipe Hands     ← 21 3-D hand landmarks at ~30 FPS
    │
    ▼
GestureDetector._classify()
    ├─ is_open_palm()        → STOP
    ├─ thumb_direction()     → FORWARD / BACKWARD
    └─ pointing_direction()  → LEFT / RIGHT
    │
    ▼
GestureResult(command, gesture_name, confidence, landmarks_detected)
```

Each detector function looks at the **relative positions of landmark y- or x-coordinates** — no ML training required beyond what MediaPipe provides out of the box.

### `app.py`

- Streamlit session state tracks `camera_running` and `current_command`.
- A `while` loop reads frames, calls `detector.process_frame()`, converts BGR → RGB, and feeds the annotated image to `st.image()`.
- The command card and ASCII robot art update on every frame rerun.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `Could not open camera index 0` | Try index `1` or `2`. On Linux, check `/dev/video*`. |
| Gestures not detected reliably | Increase lighting, reduce background clutter, lower detection confidence. |
| Low FPS | Close other camera-using apps. Reduce browser tab count. |
| `ModuleNotFoundError: mediapipe` | Ensure your virtual environment is active and you ran `pip install -r requirements.txt`. |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.8 | Webcam capture & image processing |
| `mediapipe` | ≥ 0.10 | Hand landmark detection |
| `streamlit` | ≥ 1.32 | Web UI |
| `numpy` | ≥ 1.24 | Array operations |

---

## 📄 License

MIT — free to use, modify, and distribute.
