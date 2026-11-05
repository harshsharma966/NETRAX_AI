import numpy as np
from pathlib import Path
from typing import Dict
from src.utils.logger import setup_logger

logger = setup_logger("embedding_store")


class EmbeddingStore:
    """Persists and loads gallery face embeddings (.npz file format)."""

    def __init__(self, storage_path: str = "data/face_embeddings/registered_embeddings.npz"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.load()

    def save(self):
        np.savez(str(self.storage_path), **self.embeddings)
        logger.info(f"Saved {len(self.embeddings)} face embeddings to {self.storage_path}")

    def load(self) -> Dict[str, np.ndarray]:
        if self.storage_path.exists():
            try:
                data = np.load(str(self.storage_path))
                self.embeddings = {k: data[k] for k in data.files}
                logger.info(f"Loaded {len(self.embeddings)} embeddings from store.")
            except Exception as e:
                logger.error(f"Error reading embeddings store: {e}")
                self.embeddings = {}
        return self.embeddings

    def add_user(self, name: str, embedding: np.ndarray):
        self.embeddings[name] = embedding
        self.save()

    def remove_user(self, name: str) -> bool:
        """Remove user (and any tagged variants like name#glasses) from embeddings store."""
        removed = False
        target = name.strip().lower()
        for k in list(self.embeddings.keys()):
            k_clean = k.split("#")[0].strip().lower()
            if k.lower() == target or k_clean == target:
                del self.embeddings[k]
                removed = True
        if removed:
            self.save()
            logger.info(f"Removed user '{name}' from {self.storage_path}")
        return removed
