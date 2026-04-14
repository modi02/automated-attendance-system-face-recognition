"""
Face Recognition using FaceNet embeddings (via keras-facenet or TensorFlow).
Performs distance-based similarity matching with configurable threshold.
"""

import numpy as np
import pickle
import os
from src.utils.config import Config


class FaceRecognizer:
    """
    Identifies faces by comparing 128-d FaceNet embeddings against
    a stored database using cosine or Euclidean distance.
    """

    def __init__(self, threshold: float = Config.RECOGNITION_THRESHOLD):
        self.threshold = threshold
        self.metric = Config.SIMILARITY_METRIC
        self.embeddings_db: dict = {}   # {name: [embedding, ...]}
        self._model = None
        self._load_model()
        self._load_embeddings()

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Load FaceNet model (keras-facenet preferred, TF SavedModel fallback)."""
        try:
            from keras_facenet import FaceNet
            self._model = FaceNet()
            self._backend = "keras_facenet"
            print("[INFO] FaceNet model loaded via keras-facenet.")
        except ImportError:
            try:
                import tensorflow as tf
                model_path = os.path.join(Config.MODELS_DIR, "facenet_keras.h5")
                if os.path.exists(model_path):
                    self._model = tf.keras.models.load_model(model_path, compile=False)
                    self._backend = "tensorflow"
                    print("[INFO] FaceNet model loaded via TensorFlow.")
                else:
                    print("[WARN] No FaceNet model found. Using random embeddings (demo mode).")
                    self._model = None
                    self._backend = "demo"
            except Exception as e:
                print(f"[WARN] Model load error: {e}. Running in demo mode.")
                self._model = None
                self._backend = "demo"

    # ------------------------------------------------------------------ #
    # Embedding extraction                                                 #
    # ------------------------------------------------------------------ #

    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract 128-d embedding from a face image (BGR, any size).
        Returns normalised unit vector.
        """
        import cv2
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))

        if self._backend == "keras_facenet":
            import tensorflow as tf
            embedding = self._model.embeddings(
                np.expand_dims(face_resized, axis=0)
            )[0]
        elif self._backend == "tensorflow":
            face_norm = (face_resized.astype(np.float32) - 127.5) / 128.0
            embedding = self._model.predict(np.expand_dims(face_norm, axis=0))[0]
        else:
            # Demo mode: reproducible random embedding based on pixel mean
            rng = np.random.RandomState(int(face_resized.mean() * 1000) % 2**31)
            embedding = rng.randn(Config.EMBEDDING_DIM).astype(np.float32)

        # L2 normalise
        norm = np.linalg.norm(embedding)
        return embedding / (norm + 1e-10)

    # ------------------------------------------------------------------ #
    # Similarity & identification                                          #
    # ------------------------------------------------------------------ #

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            return float(1.0 - np.dot(a, b))          # both are unit vectors
        return float(np.linalg.norm(a - b))            # Euclidean

    def identify(self, face_img: np.ndarray) -> tuple[str, float]:
        """
        Given a face crop, return (name, confidence_score).
        confidence_score is the similarity distance (lower = more confident).
        """
        if not self.embeddings_db:
            return "Unknown", 1.0

        query = self.get_embedding(face_img)
        best_name, best_dist = "Unknown", self.threshold

        for name, embeddings in self.embeddings_db.items():
            dists = [self._distance(query, emb) for emb in embeddings]
            min_dist = min(dists)
            if min_dist < best_dist:
                best_dist = min_dist
                best_name = name

        return best_name, best_dist

    # ------------------------------------------------------------------ #
    # Embedding persistence                                                #
    # ------------------------------------------------------------------ #

    def _load_embeddings(self):
        if os.path.exists(Config.EMBEDDINGS_FILE):
            with open(Config.EMBEDDINGS_FILE, "rb") as f:
                self.embeddings_db = pickle.load(f)
            print(f"[INFO] Loaded embeddings for {len(self.embeddings_db)} student(s).")
        else:
            print("[INFO] No embeddings found. Register students first.")

    def save_embeddings(self):
        with open(Config.EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(self.embeddings_db, f)
        print(f"[INFO] Embeddings saved → {Config.EMBEDDINGS_FILE}")

    def add_student(self, name: str, embeddings: list[np.ndarray]):
        """Add or update a student's embeddings in the database."""
        self.embeddings_db[name] = embeddings
        self.save_embeddings()
        print(f"[INFO] Registered student: {name} ({len(embeddings)} embedding(s)).")

    def remove_student(self, name: str) -> bool:
        if name in self.embeddings_db:
            del self.embeddings_db[name]
            self.save_embeddings()
            return True
        return False

    def list_students(self) -> list[str]:
        return list(self.embeddings_db.keys())
