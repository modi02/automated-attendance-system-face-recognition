"""
Central configuration for the attendance system.
"""
import os


class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
    ATTENDANCE_DIR = os.path.join(BASE_DIR, "data", "attendance_logs")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "snapshots")
    EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "student_embeddings.pkl")

    # Face detection (YuNet)
    YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    DETECT_SCORE_THRESHOLD = 0.7
    DETECT_NMS_THRESHOLD = 0.3
    DETECT_TOP_K = 5000

    # Face recognition (FaceNet / SFace)
    RECOGNITION_THRESHOLD = 0.6          # L2 distance — lower = stricter
    EMBEDDING_DIM = 128                   # FaceNet 128-d embeddings
    SIMILARITY_METRIC = "cosine"          # "cosine" or "euclidean"

    # Attendance
    ATTENDANCE_COOLDOWN_SECONDS = 300     # Re-mark same student after 5 min
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:%S"
    CSV_DATE_COLUMN = "Date"
    CSV_NAME_COLUMN = "Name"
    CSV_TIME_COLUMN = "Time"
    CSV_STATUS_COLUMN = "Status"

    # Camera
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30

    # Ensure directories exist
    for d in [EMBEDDINGS_DIR, ATTENDANCE_DIR, MODELS_DIR, SNAPSHOTS_DIR]:
        os.makedirs(d, exist_ok=True)
