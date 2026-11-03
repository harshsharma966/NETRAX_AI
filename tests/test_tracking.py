try:
    import pytest
except ImportError:
    pytest = None

import unittest
from src.tracking.person_tracker import PersonTracker


class TestPersonTracker(unittest.TestCase):
    def test_person_tracker_assignment(self):
        tracker = PersonTracker()
        detections_frame1 = [{"bbox": [50, 50, 100, 150], "confidence": 0.9}]
        tracks1 = tracker.update(detections_frame1)
        self.assertEqual(len(tracks1), 1)
        self.assertEqual(tracks1[0]["track_id"], 1)

        # Second frame with slight shift
        detections_frame2 = [{"bbox": [55, 55, 105, 155], "confidence": 0.88}]
        tracks2 = tracker.update(detections_frame2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0]["track_id"], 1)

    def test_bytetrack_low_confidence_association(self):
        tracker = PersonTracker()
        # Initial high-confidence detection
        tracker.update([{"bbox": [100, 100, 200, 300], "confidence": 0.92}])

        # Occluded / blurry frame with low confidence (0.3)
        tracks = tracker.update([{"bbox": [105, 105, 205, 305], "confidence": 0.30}])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["track_id"], 1)


def test_person_tracker_assignment():
    test_case = TestPersonTracker()
    test_case.test_person_tracker_assignment()


def test_bytetrack_low_confidence_association():
    test_case = TestPersonTracker()
    test_case.test_bytetrack_low_confidence_association()


if __name__ == "__main__":
    unittest.main()
