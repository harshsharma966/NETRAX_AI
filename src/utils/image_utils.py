import cv2
import numpy as np
from typing import Tuple, Optional


def safe_crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Safely crop an image region with bounding box clipping.
    
    Args:
        image: Source image numpy array (H, W, C)
        bbox: Bounding box tuple (x1, y1, x2, y2)
    """
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw labeled bounding box on an image."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        font_scale = 0.5
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA
        )
    return image
