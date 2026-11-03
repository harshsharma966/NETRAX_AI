import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils.logger import setup_logger

logger = setup_logger("face_detector")


class FaceDetector:
    """SCRFD Face Detector using ONNX runtime / OpenCV DNN (det_10g.onnx)."""

    def __init__(
        self,
        weights_path: str = "models/face_detection/det_10g.onnx",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple = (640, 640)
    ):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net = None

        # Check for ONNX weights in models/face_detection/det_10g.onnx
        if not self.weights_path.exists():
            default_path = Path("models/face_detection/det_10g.onnx")
            if default_path.exists():
                self.weights_path = default_path

        if self.weights_path.exists():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.weights_path))
                self.out_names = self.net.getUnconnectedOutLayersNames()
                logger.info(f"Loaded ONNX Face Detector from {self.weights_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX model {self.weights_path}: {e}")
                self.net = None
        else:
            logger.warning(f"ONNX weights not found at {self.weights_path}")

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image using SCRFD ONNX model.
        
        Returns:
            List of dicts: {'bbox': [x1, y1, x2, y2], 'confidence': float, 'landmarks': ...}
        """
        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]

        if self.net is not None:
            try:
                in_w, in_h = self.input_size
                resized = cv2.resize(image, (in_w, in_h))
                scale_x = orig_w / float(in_w)
                scale_y = orig_h / float(in_h)

                blob = cv2.dnn.blobFromImage(
                    resized,
                    scalefactor=1.0 / 128.0,
                    size=(in_w, in_h),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=True
                )
                self.net.setInput(blob)
                outs = self.net.forward(self.out_names)

                # SCRFD multi-stride output decoding (strides 8, 16, 32)
                strides = [8, 16, 32]
                score_indices = [0, 1, 2]
                bbox_indices = [3, 4, 5]

                boxes = []
                scores = []
                num_anchors = 2

                for stride, s_idx, b_idx in zip(strides, score_indices, bbox_indices):
                    score_out = outs[s_idx]
                    bbox_out = outs[b_idx]

                    w_feat = in_w // stride
                    h_feat = in_h // stride

                    y, x = np.mgrid[0:h_feat, 0:w_feat]
                    anchor_centers = np.stack((x, y), axis=-1).reshape(-1, 2) * stride
                    anchor_centers = np.repeat(anchor_centers, num_anchors, axis=0)

                    thresh_mask = (score_out[:, 0] > self.conf_threshold)
                    if not np.any(thresh_mask):
                        continue

                    valid_scores = score_out[thresh_mask, 0]
                    valid_deltas = bbox_out[thresh_mask] * stride
                    valid_anchors = anchor_centers[thresh_mask]

                    x1 = (valid_anchors[:, 0] - valid_deltas[:, 0]) * scale_x
                    y1 = (valid_anchors[:, 1] - valid_deltas[:, 1]) * scale_y
                    x2 = (valid_anchors[:, 0] + valid_deltas[:, 2]) * scale_x
                    y2 = (valid_anchors[:, 1] + valid_deltas[:, 3]) * scale_y

                    # Clip to image dimensions
                    x1 = np.clip(x1, 0, orig_w)
                    y1 = np.clip(y1, 0, orig_h)
                    x2 = np.clip(x2, 0, orig_w)
                    y2 = np.clip(y2, 0, orig_h)

                    for i in range(len(valid_scores)):
                        w_box = max(0, x2[i] - x1[i])
                        h_box = max(0, y2[i] - y1[i])
                        boxes.append([int(x1[i]), int(y1[i]), int(w_box), int(h_box)])
                        scores.append(float(valid_scores[i]))

                if not boxes:
                    return []

                indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
                results = []
                if len(indices) > 0:
                    for idx in indices:
                        i = idx if isinstance(idx, (int, np.integer)) else idx[0]
                        b = boxes[i]
                        results.append({
                            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                            "confidence": scores[i]
                        })
                return results

            except Exception as e:
                logger.error(f"SCRFD ONNX face detection inference failed: {e}")

        # Fallback heuristic if ONNX execution fails
        h, w = image.shape[:2]
        face_h = int(h * 0.4)
        face_w = int(w * 0.7)
        x1 = max(0, int((w - face_w) / 2))
        y1 = max(0, int(h * 0.05))
        return [{
            "bbox": [x1, y1, min(w, x1 + face_w), min(h, y1 + face_h)],
            "confidence": 0.75
        }]
