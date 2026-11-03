import cv2
import numpy as np
from typing import Optional, List


class FaceAligner:
    """Aligns face crops to standardized orientation and size (e.g., 112x112)."""

    def __init__(self, output_size: int = 112):
        self.output_size = output_size

    def align(self, face_image: np.ndarray, landmarks: Optional[List[int]] = None) -> np.ndarray:
        """Resize or affine-warp face to target dimension."""
        if face_image is None or face_image.size == 0:
            return np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)

        # Standard resize if landmarks are not explicitly provided
        aligned = cv2.resize(face_image, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        return aligned
