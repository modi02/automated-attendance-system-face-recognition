"""
Offline Student Registration Module.
Captures multiple face samples from webcam, extracts embeddings, and stores them.
"""

import cv2
import numpy as np
import time
from src.recognition.face_detector import FaceDetector
from src.recognition.face_recognizer import FaceRecognizer
from src.utils.config import Config


class StudentRegistration:
    """
    Interactive CLI + webcam tool to register new students.
    Captures N face samples, computes embeddings, and persists to pickle.
    """

    SAMPLES_REQUIRED = 10       # Number of face samples to capture per student
    CAPTURE_INTERVAL = 0.5      # Seconds between captures

    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer):
        self.detector = detector
        self.recognizer = recognizer

    def run(self):
        print("\n" + "=" * 50)
        print("  STUDENT REGISTRATION MODE")
        print("=" * 50)
        existing = self.recognizer.list_students()
        if existing:
            print(f"  Registered students ({len(existing)}): {', '.join(existing)}")
        print()

        while True:
            name = input("Enter student name (or 'quit' to exit): ").strip()
            if name.lower() == "quit":
                break
            if not name:
                print("[WARN] Name cannot be empty.")
                continue

            print(f"\n[INFO] Registering '{name}'. Look at the camera...")
            embeddings = self._capture_embeddings(name)

            if embeddings:
                self.recognizer.add_student(name, embeddings)
                print(f"[OK] '{name}' registered successfully with {len(embeddings)} samples.\n")
            else:
                print(f"[FAIL] Registration failed for '{name}'. Try again.\n")

    def _capture_embeddings(self, name: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return []

        embeddings = []
        last_capture = 0
        print(f"  Capturing {self.SAMPLES_REQUIRED} samples. Hold still...")

        while len(embeddings) < self.SAMPLES_REQUIRED:
            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()
            faces = self.detector.detect(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                now = time.time()
                if now - last_capture >= self.CAPTURE_INTERVAL and len(embeddings) < self.SAMPLES_REQUIRED:
                    face_crop = frame[y:y + h, x:x + w]
                    if face_crop.size > 0:
                        emb = self.recognizer.get_embedding(face_crop)
                        embeddings.append(emb)
                        last_capture = now
                        print(f"  Sample {len(embeddings)}/{self.SAMPLES_REQUIRED} captured.")

            # Progress bar overlay
            progress = int((len(embeddings) / self.SAMPLES_REQUIRED) * frame.shape[1])
            cv2.rectangle(display, (0, frame.shape[0] - 20), (progress, frame.shape[0]), (0, 220, 0), -1)
            cv2.putText(display, f"Registering: {name}  [{len(embeddings)}/{self.SAMPLES_REQUIRED}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Registration", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return embeddings
