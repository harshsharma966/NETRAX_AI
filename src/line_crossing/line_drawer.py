import cv2
import numpy as np
from typing import Tuple


class LineDrawer:
    """Draws virtual tripwire lines and crossing feedback on images."""

    @staticmethod
    def draw_line(
        image: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 3,
        label: str = "ENTRY LINE"
    ) -> np.ndarray:
        cv2.line(image, pt1, pt2, color, thickness)
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2 - 10
        cv2.putText(
            image,
            label,
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )
        return image
