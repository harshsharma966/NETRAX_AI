import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger("person_detector")


class PersonDetector:
    """Person detector utilizing YOLO11/YOLO architecture."""

    def __init__(
        self,
        weights_path: str = "models/person_detection/yolo11n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            if not self.weights_path.exists() or self.weights_path.stat().st_size < 1000:
                logger.warning(f"Weights file {self.weights_path} not ready. Loading YOLO('yolo11n.pt') directly.")
                self.model = YOLO("yolo11n.pt")
            else:
                self.model = YOLO(str(self.weights_path))
            self.model.to(self.device)
            logger.info("PersonDetector loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons in a given frame with fast imgsz=416."""
        if self.model is None or frame is None:
            return []

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],
            imgsz=416,
            verbose=False,
            device=self.device
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append({
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": conf,
                    "class_id": cls_id
                })
        return detections
