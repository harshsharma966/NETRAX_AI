import cv2
import numpy as np
from typing import Tuple, Dict, Any


class FaceQualityChecker:
    """Evaluates blurriness, resolution, and brightness to guarantee high-quality embeddings."""

    def __init__(self, min_size: Tuple[int, int] = (60, 60), min_blur_score: float = 50.0):
        self.min_w, self.min_h = min_size
        self.min_blur_score = min_blur_score

    def check_quality(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """Compute quality metrics for a face crop."""
        if face_crop is None or face_crop.size == 0:
            return {"is_valid": False, "reason": "empty_crop", "score": 0.0}

        h, w = face_crop.shape[:2]
        if w < self.min_w or h < self.min_h:
            return {"is_valid": False, "reason": "too_small", "score": float(w * h)}

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)

        is_valid = (blur_score >= self.min_blur_score) and (40 <= brightness <= 220)
        return {
            "is_valid": is_valid,
            "blur_score": float(blur_score),
            "brightness": float(brightness),
            "resolution": (w, h)
        }




# 5. Blur Detection – Most Important
# blur_score = cv2.Laplacian(    gray,    cv2.CV_64F).var()



# Ye code Variance of Laplacian technique use karta hai.

# Iska purpose image ke edges aur sharpness ko measure karna hai.
# - Sharp image: Strong edges → Generally higher variance.
# - Blurry image: Smooth edges → Generally lower variance.