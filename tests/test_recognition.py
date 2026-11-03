import pytest
import numpy as np
from src.recognition.face_matcher import FaceMatcher


def test_face_matcher_cosine():
    matcher = FaceMatcher(threshold=0.7)
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    gallery = {"Alice": v1, "Bob": v3}
    name, score = matcher.match(v2, gallery)
    assert name == "Alice"
    assert pytest.approx(score, 0.01) == 1.0
