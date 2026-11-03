import cv2
import torch
import numpy as np
from src.recognition.model import load_recognition_model, FaceRecognitionBackbone


class EmbeddingExtractor:
    """Extracts L2-normalized 512-D embedding vectors from aligned face images."""

    def __init__(self, weights_path: str = "models/face_recognition/best_model.pth", device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model: FaceRecognitionBackbone = load_recognition_model(weights_path, str(self.device))

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding from an aligned 112x112 face image."""
        if face_image is None or face_image.size == 0:
            return np.zeros((512,), dtype=np.float32)

        # Preprocessing: BGR -> RGB, standard normalization [-1, 1]
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0)
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)
            emb_np = emb.squeeze(0).cpu().numpy()

        return emb_np
