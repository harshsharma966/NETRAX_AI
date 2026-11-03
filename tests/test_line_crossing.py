import pytest
from src.line_crossing.crossing_detector import CrossingDetector


def test_line_crossing_event():
    # Tripwire horizontal line at y=100 from x=0 to x=200
    detector = CrossingDetector(line_start=(0, 100), line_end=(200, 100), entry_direction="top_to_bottom")
    trajectory = [(50, 80), (50, 120)]
    event = detector.check_crossing(track_id=1, trajectory=trajectory)
    assert event == "ENTRY"
