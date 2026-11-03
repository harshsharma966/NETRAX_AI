from typing import Tuple, List, Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger("crossing_detector")


class CrossingDetector:
    """Detects when a tracked person crosses a configured tripwire line."""

    def __init__(
        self,
        line_start: Tuple[int, int] = (100, 350),
        line_end: Tuple[int, int] = (1180, 350),
        entry_direction: str = "top_to_bottom"
    ):
        self.line_start = line_start
        self.line_end = line_end
        self.entry_direction = entry_direction
        self.crossed_ids = set()

    @staticmethod
    def _orientation(p1: Tuple[int, int], p2: Tuple[int, int], p: Tuple[int, int]) -> float:
        """Cross product to determine side of point p relative to vector p1->p2."""
        return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    def check_crossing(self, track_id: int, trajectory: List[Tuple[int, int]]) -> Optional[str]:
        """Determines if a track trajectory crosses the tripwire line.
        
        Returns:
            'ENTRY', 'EXIT', or None.
        """
        if track_id in self.crossed_ids or len(trajectory) < 2:
            return None

        p_prev = trajectory[-2]
        p_curr = trajectory[-1]

        side_prev = self._orientation(self.line_start, self.line_end, p_prev)
        side_curr = self._orientation(self.line_start, self.line_end, p_curr)

        # A sign change indicates crossing the line segment
        if (side_prev > 0 and side_curr < 0) or (side_prev < 0 and side_curr > 0):
            # Evaluate direction
            dy = p_curr[1] - p_prev[1]
            dx = p_curr[0] - p_prev[0]

            is_entry = False
            if self.entry_direction == "top_to_bottom" and dy > 0:
                is_entry = True
            elif self.entry_direction == "bottom_to_top" and dy < 0:
                is_entry = True
            elif self.entry_direction == "left_to_right" and dx > 0:
                is_entry = True
            elif self.entry_direction == "right_to_left" and dx < 0:
                is_entry = True

            event_type = "ENTRY" if is_entry else "EXIT"
            self.crossed_ids.add(track_id)
            logger.info(f"Track {track_id} triggered {event_type} event.")
            return event_type

        return None
