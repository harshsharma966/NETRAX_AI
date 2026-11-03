import cv2
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from src.detection.person_detector import PersonDetector
from src.tracking.person_tracker import PersonTracker
from src.line_crossing.line_drawer import LineDrawer
from src.line_crossing.crossing_detector import CrossingDetector
from src.face.face_capture import FaceCaptureManager
from src.recognition.embedding_extractor import EmbeddingExtractor
from src.recognition.face_matcher import FaceMatcher
from src.database.db_connection import DBConnection
from src.database.embedding_store import EmbeddingStore
from src.database.entry_logs import EntryLogger
from src.utils.image_utils import safe_crop, draw_bounding_box
from src.utils.logger import setup_logger

logger = setup_logger("pipeline")


class PersonEntryPipeline:
    """End-to-end pipeline uniting detection, tracking, crossing, and face recognition."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        # Components
        m_cfg = self.cfg.get("models", {})
        self.detector = PersonDetector(
            weights_path=m_cfg.get("person_detection", {}).get("weights", "models/person_detection/yolo11n.pt"),
            conf_threshold=m_cfg.get("person_detection", {}).get("confidence_threshold", 0.5),
            device=m_cfg.get("person_detection", {}).get("device", "cpu")
        )
        tr_cfg = self.cfg.get("tracking", {})
        self.tracker = PersonTracker(
            track_high_thresh=tr_cfg.get("track_high_thresh", 0.5),
            track_low_thresh=tr_cfg.get("track_low_thresh", 0.1),
            new_track_thresh=tr_cfg.get("new_track_thresh", 0.6),
            match_thresh=tr_cfg.get("match_thresh", 0.8),
            track_buffer=tr_cfg.get("track_buffer", 90)
        )

        lc_cfg = self.cfg.get("line_crossing", {})
        self.crossing_detector = CrossingDetector(
            line_start=tuple(lc_cfg.get("line_start", [100, 350])),
            line_end=tuple(lc_cfg.get("line_end", [1180, 350])),
            entry_direction=lc_cfg.get("entry_direction", "top_to_bottom")
        )
        self.line_start = tuple(lc_cfg.get("line_start", [100, 350]))
        self.line_end = tuple(lc_cfg.get("line_end", [1180, 350]))

        self.face_capture = FaceCaptureManager(output_dir=self.cfg.get("app", {}).get("captures_dir", "captures"))
        self.extractor = EmbeddingExtractor(
            weights_path=m_cfg.get("face_recognition", {}).get("weights", "models/face_recognition/iresnet100.onnx"),
            device=m_cfg.get("face_recognition", {}).get("device", "cpu")
        )
        self.matcher = FaceMatcher(
            threshold=m_cfg.get("face_recognition", {}).get("similarity_threshold", 0.40)
        )

        # Database — PostgreSQL only, config from .env
        db_cfg = self.cfg.get("database", {})
        self.db = DBConnection()
        self.emb_store = EmbeddingStore(storage_path=db_cfg.get("embeddings_path", "data/face_embeddings/registered_embeddings.npz"))
        self.entry_logger = EntryLogger(self.db)

        # Recognized track maps
        self.recognized_names: Dict[int, str] = {}
        self.recognized_scores: Dict[int, float] = {}
        self.frame_count: int = 0

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """Process single frame with optimized detection and throttled face recognition."""
        self.frame_count += 1
        display = frame.copy()

        # 1. Person Detection
        detections = self.detector.detect(frame)

        # 2. Tracking
        tracks = self.tracker.update(detections)

        # Draw virtual tripwire line
        LineDrawer.draw_line(display, self.line_start, self.line_end)

        for tr in tracks:
            tid = tr["track_id"]
            bbox = tr["bbox"]
            conf = tr["confidence"]

            # Crop person region for face capture
            person_crop = safe_crop(frame, bbox)
            if person_crop is not None:
                self.face_capture.process_person_crop(tid, person_crop)

            # Throttled Face Recognition (avoids CPU lag by not re-running 100-layer net every frame)
            need_recognition = (
                tid not in self.recognized_names
                or (self.frame_count % 10 == 0 and self.recognized_names.get(tid) == "Unknown")
            )

            if need_recognition:
                best_face = self.face_capture.get_best_face(tid)
                if best_face is not None:
                    emb = self.extractor.extract(best_face)
                    name, score = self.matcher.match(emb, self.emb_store.embeddings)
                    self.recognized_names[tid] = name
                    self.recognized_scores[tid] = score

            # 3. Line crossing verification
            history = self.tracker.tracks[tid]["history"]
            crossing_event = self.crossing_detector.check_crossing(tid, history)

            if crossing_event:
                name = self.recognized_names.get(tid, "Unknown")
                score = self.recognized_scores.get(tid, 0.0)

                # Save snapshot
                snap_path = f"captures/person_crops/track_{tid}_{name}.jpg"
                if person_crop is not None:
                    cv2.imwrite(snap_path, person_crop)

                # Log into database
                self.entry_logger.log_entry(
                    track_id=tid,
                    person_name=name,
                    confidence=score,
                    direction=crossing_event,
                    snapshot_path=snap_path,
                )

            # Labeling and color indicators
            label_name = self.recognized_names.get(tid, None)
            score = self.recognized_scores.get(tid, None)

            if label_name is not None and label_name != "Unknown":
                # Registered Person: Green box
                display_label = f"{label_name} ({score:.2f})"
                box_color = (0, 255, 0)
            elif label_name == "Unknown":
                # Detected Face but Unregistered: Orange/Red box
                display_label = "Unknown"
                box_color = (0, 140, 255)
            else:
                # Face not yet clearly visible: Cyan tracking ID
                display_label = f"ID:{tid}"
                box_color = (255, 200, 0)

            draw_bounding_box(display, bbox, label=display_label, color=box_color)

        return display
