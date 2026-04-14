"""
Face Detection using YuNet (OpenCV DNN-based detector).
Supports webcam and 5G IP camera streams.
"""

import cv2
import numpy as np
from src.utils.config import Config
import os


class FaceDetector:
    """
    Wraps OpenCV's YuNet face detector for reliable multi-face detection.
    Falls back to Haar cascade if the ONNX model is not available.
    """

    def __init__(self):
        self._use_yunet = os.path.exists(Config.YUNET_MODEL_PATH)
        if self._use_yunet:
            self.detector = cv2.FaceDetectorYN.create(
                Config.YUNET_MODEL_PATH,
                "",
                (320, 320),
                Config.DETECT_SCORE_THRESHOLD,
                Config.DETECT_NMS_THRESHOLD,
                Config.DETECT_TOP_K,
            )
            print("[INFO] YuNet detector loaded.")
        else:
            print("[WARN] YuNet model not found. Using Haar cascade fallback.")
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect faces in a BGR frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if self._use_yunet:
            return self._detect_yunet(frame)
        return self._detect_haar(frame)

    def _detect_yunet(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        boxes = []
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # Clamp to frame boundaries
            x, y = max(0, x), max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)
            if fw > 0 and fh > 0:
                boxes.append((x, y, fw, fh))
        return boxes

    def _detect_haar(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return [(x, y, w, h) for (x, y, w, h) in faces] if len(faces) else []

    def align_face(self, frame: np.ndarray, box: tuple, target_size=(160, 160)) -> np.ndarray:
        """Crop and resize face region for embedding extraction."""
        x, y, w, h = box
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            return np.zeros((*target_size, 3), dtype=np.uint8)
        return cv2.resize(face, target_size)
