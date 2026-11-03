import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from src.face.face_detector import FaceDetector
from src.face.face_quality import FaceQualityChecker
from src.face.face_alignment import FaceAligner
from src.utils.image_utils import safe_crop


class FaceCaptureManager:
    """Buffers and selects the highest quality face image per track ID."""

    def __init__(self, output_dir: str = "captures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector = FaceDetector()
        self.quality_checker = FaceQualityChecker()
        self.aligner = FaceAligner()
        # track_id -> {'best_face': img, 'best_score': float, 'best_crop': img}
        self.track_faces: Dict[int, Dict[str, Any]] = {}

    def process_person_crop(self, track_id: int, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """Search for face in person crop, assess quality, and update best snapshot."""
        faces = self.detector.detect_faces(person_crop)
        if not faces:
            return None

        # Take largest detected face
        face_info = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        face_crop = safe_crop(person_crop, face_info["bbox"])
        if face_crop is None:
            return None

        q = self.quality_checker.check_quality(face_crop)
        score = q.get("blur_score", 0.0)

        if track_id not in self.track_faces or score > self.track_faces[track_id]["best_score"]:
            aligned = self.aligner.align(face_crop)
            self.track_faces[track_id] = {
                "best_face": aligned,
                "best_crop": face_crop,
                "best_score": score
            }
            return aligned

        return self.track_faces[track_id]["best_face"]

    def get_best_face(self, track_id: int) -> Optional[np.ndarray]:
        if track_id in self.track_faces:
            return self.track_faces[track_id]["best_face"]
        return None
