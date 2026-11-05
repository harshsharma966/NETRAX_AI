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
        # track_id -> {'best_face': img, 'current_face': img, 'best_score': float, 'last_seen_frame': int}
        self.track_faces: Dict[int, Dict[str, Any]] = {}

    def process_person_crop(self, track_id: int, person_crop: np.ndarray, frame_num: int = 0) -> Optional[np.ndarray]:
        """Search for face in person crop, assess quality, and update snapshot."""
        faces = self.detector.detect_faces(person_crop)
        if not faces:
            return None

        # Take largest detected face
        face_info = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        landmarks = face_info.get("landmarks")

        # 5-Point Affine alignment: rotates tilted head upright directly into 112x112
        if landmarks is not None:
            aligned = self.aligner.align(person_crop, landmarks=landmarks)
            face_crop = safe_crop(person_crop, face_info["bbox"])
            if face_crop is None:
                face_crop = person_crop
        else:
            face_crop = safe_crop(person_crop, face_info["bbox"])
            if face_crop is None:
                return None
            aligned = self.aligner.align(face_crop)

        q = self.quality_checker.check_quality(face_crop)
        score = q.get("blur_score", 0.0)

        if track_id not in self.track_faces:
            self.track_faces[track_id] = {
                "best_face": aligned,
                "current_face": aligned,
                "best_score": score,
                "last_seen_frame": frame_num
            }
        else:
            self.track_faces[track_id]["current_face"] = aligned
            self.track_faces[track_id]["last_seen_frame"] = frame_num
            # If current face quality is better or old face is older than 30 frames, update best_face
            if score > self.track_faces[track_id]["best_score"] or (frame_num - self.track_faces[track_id]["last_seen_frame"] > 30):
                self.track_faces[track_id]["best_face"] = aligned
                self.track_faces[track_id]["best_score"] = score

        return aligned

    def get_best_face(self, track_id: int) -> Optional[np.ndarray]:
        if track_id in self.track_faces:
            current = self.track_faces[track_id].get("current_face")
            if current is not None:
                return current
            return self.track_faces[track_id].get("best_face")
        return None

    def clear_inactive_tracks(self, active_track_ids: set, current_frame: int, max_idle: int = 45):
        """Remove cached faces for tracks that have left the screen."""
        stale_ids = [
            tid for tid, data in self.track_faces.items()
            if (tid not in active_track_ids and current_frame - data.get("last_seen_frame", 0) > max_idle)
        ]
        for tid in stale_ids:
            self.track_faces.pop(tid, None)
