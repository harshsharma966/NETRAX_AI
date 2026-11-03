import pytest
import numpy as np
from src.detection.person_detector import PersonDetector


def test_person_detector_initialization():
    detector = PersonDetector(weights_path="models/person_detection/yolo11n.pt", device="cpu")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list)
