# 🎓 Automated Attendance System using Face Recognition

A real-time, offline-capable attendance management system powered by **YuNet** face detection and **FaceNet** embeddings. Supports webcam and 5G IP camera streams, with automated daily CSV logging via Pandas.

> **Tools & Technologies:** Python · TensorFlow · Keras-FaceNet · OpenCV · Pandas

---

## ✨ Features

### 🔍 Real-Time Face Recognition
- **YuNet** (OpenCV DNN) for fast, accurate multi-face detection — compatible with webcam and 5G IP camera RTSP streams
- **FaceNet** 128-d embeddings extracted via `keras-facenet` for identity recognition
- **Distance-based similarity matching** with configurable threshold (cosine or Euclidean)
- Falls back to Haar cascade if the ONNX model is unavailable

### 🧑‍🎓 Offline Student Registration
- Interactive webcam-based registration — captures 10 face samples per student
- Embeddings stored persistently as **pickle** files (no cloud dependency)
- CLI tools to list and remove registered students

### 📋 Automated Attendance Logging
- Marks attendance on recognition with a configurable **cooldown period** (default: 5 min) to prevent duplicates
- Records: Date, Name, Time, Status — saved to dated CSV files
- Live on-screen counter showing students marked present

### 📊 Report Generation
- Daily attendance reports via CLI
- 7-day pivot-table summary (student × date)
- Excel export support via `openpyxl`

---

## 🏗️ Architecture

```
main.py
  ├── FaceDetector          — YuNet / Haar face bounding boxes
  ├── FaceRecognizer        — FaceNet embeddings + distance matching
  │     └── EmbeddingDB     — Pickle-persisted student database
  └── AttendanceLogger      — Cooldown logic + Pandas CSV I/O

register.py                 — Offline student registration CLI
report.py                   — Report generation & Excel export
```

### Recognition Pipeline

```
Video Frame
    │
    ▼
YuNet Face Detection  ──►  Bounding boxes [(x,y,w,h), ...]
    │
    ▼
FaceNet Embedding     ──►  128-d unit vector per face
    │
    ▼
Cosine Similarity     ──►  Compare against stored embeddings
    │
    ▼
Threshold Check       ──►  distance < 0.6 → Identified
    │
    ▼
AttendanceLogger      ──►  Mark → Save to CSV
```

---

## 📁 Project Structure

```
automated-attendance-system-face-recognition/
├── main.py                       # Entry point — live recognition loop
├── register.py                   # Student registration CLI
├── report.py                     # Report generation script
├── requirements.txt
├── src/
│   ├── recognition/
│   │   ├── face_detector.py      # YuNet / Haar wrapper
│   │   ├── face_recognizer.py    # FaceNet embeddings + similarity matching
│   │   └── student_registration.py  # Webcam capture + embedding storage
│   ├── attendance/
│   │   └── attendance_logger.py  # Cooldown logic, CSV I/O, reporting
│   └── utils/
│       └── config.py             # Centralized path and parameter config
├── data/
│   ├── embeddings/               # student_embeddings.pkl (gitignored)
│   ├── attendance_logs/          # attendance_YYYY-MM-DD.csv (gitignored)
│   └── snapshots/
└── models/                       # YuNet ONNX model (download separately)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Webcam or IP camera with RTSP stream
- (Optional) GPU with CUDA for faster TensorFlow inference

### 1. Clone the repository
```bash
git clone https://github.com/your-username/automated-attendance-system-face-recognition.git
cd automated-attendance-system-face-recognition
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the YuNet model
```bash
mkdir -p models
wget -O models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

---

## 🖥️ Usage

### Step 1 — Register students
```bash
python register.py
```
Follow the prompts. Each student requires ~5 seconds in front of the camera.

```bash
# List registered students
python register.py --list

# Remove a student
python register.py --remove "John Doe"
```

### Step 2 — Run the attendance system

```bash
# Webcam (default)
python main.py

# Specific webcam index
python main.py --source 1

# IP / 5G camera via RTSP
python main.py --source "rtsp://192.168.1.100:554/stream"

# Custom recognition threshold
python main.py --threshold 0.5
```

**Controls:**
- `q` — quit
- `s` — save snapshot

### Step 3 — Generate reports

```bash
# Today's report
python report.py

# Specific date
python report.py --date 2025-10-15

# 7-day summary table
python report.py --summary

# Export summary to Excel
python report.py --export
```

---

## ⚙️ Configuration

All parameters are in `src/utils/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `RECOGNITION_THRESHOLD` | `0.6` | Max cosine distance for a match |
| `ATTENDANCE_COOLDOWN_SECONDS` | `300` | Seconds before re-marking same student |
| `SAMPLES_REQUIRED` | `10` | Face samples per student during registration |
| `SIMILARITY_METRIC` | `cosine` | `cosine` or `euclidean` |
| `DETECT_SCORE_THRESHOLD` | `0.7` | YuNet detection confidence threshold |

---

## 📊 Sample Attendance CSV

```csv
Date,Name,Time,Status
2025-10-15,Alice Johnson,09:02:14,Present
2025-10-15,Bob Smith,09:05:31,Present
2025-10-15,Carol White,09:11:08,Present
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `YuNet model not found` | Download the ONNX file (see Step 4) — system falls back to Haar cascade |
| `keras_facenet not found` | `pip install keras-facenet` or use TF SavedModel |
| Low recognition accuracy | Lower `RECOGNITION_THRESHOLD` (e.g. 0.5) or capture more registration samples |
| RTSP stream not opening | Verify camera IP, port, and credentials; install `ffmpeg` |

---

## 📸 Screenshots

> Add screenshots of your running system here.

---

## 📄 License

MIT © 2025
