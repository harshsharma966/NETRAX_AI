import numpy as np
from typing import Dict, Tuple, Optional


class FaceMatcher:
    """Matches query face embeddings against stored gallery embeddings using Cosine Similarity."""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def match(self, query_emb: np.ndarray, gallery: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """Find highest matching identity in the gallery.
        
        Returns:
            Tuple of (identity_name, similarity_score). Returns ('Unknown', score) if below threshold.
        """
        if not gallery or query_emb is None:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0

        for name, emb in gallery.items():
            sim = self.cosine_similarity(query_emb, emb)
            if sim > best_score:
                best_score = sim
                clean_name = name.split("#")[0]
                best_name = clean_name

        if best_score >= self.threshold:
            return best_name, best_score
        return "Unknown", max(0.0, best_score)
