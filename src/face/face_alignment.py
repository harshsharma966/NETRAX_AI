import cv2
import numpy as np
from typing import Optional, List

# Standard InsightFace ArcFace 5-point reference landmarks for 112x112 target
ARCFACE_DST_PTS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)


class FaceAligner:
    """Aligns face crops to standardized orientation and size (112x112) using 5-point affine warping."""

    def __init__(self, output_size: int = 112):
        self.output_size = output_size
        self.dst_pts = ARCFACE_DST_PTS.copy()
        if output_size != 112:
            self.dst_pts = self.dst_pts * (output_size / 112.0)

    def align(self, face_image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Affine-warp face using 5 facial landmarks to rotate and align head straight."""
        if face_image is None or face_image.size == 0:
            return np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)

        if landmarks is not None and len(landmarks) == 5:
            try:
                src_pts = np.asarray(landmarks, dtype=np.float32)
                # Compute optimal similarity transform (rotation + scale + translation)
                M, _ = cv2.estimateAffinePartial2D(src_pts, self.dst_pts, method=cv2.LMEDS)
                if M is not None:
                    aligned = cv2.warpAffine(
                        face_image,
                        M,
                        (self.output_size, self.output_size),
                        borderValue=0.0,
                        flags=cv2.INTER_LINEAR
                    )
                    return aligned
            except Exception:
                pass

        # Standard resize fallback if landmarks are not available or alignment fails
        aligned = cv2.resize(face_image, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        return aligned
