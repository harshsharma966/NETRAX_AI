import cv2
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Set

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
            track_buffer=tr_cfg.get("track_buffer", 30)
        )

        lc_cfg = self.cfg.get("line_crossing", {})
        self.line_crossing_enabled = lc_cfg.get("enabled", False)
        self.crossing_detector = CrossingDetector(
            line_start=tuple(lc_cfg.get("line_start", [100, 350])),
            line_end=tuple(lc_cfg.get("line_end", [1180, 350])),
            entry_direction=lc_cfg.get("entry_direction", "top_to_bottom")
        )
        self.line_start = tuple(lc_cfg.get("line_start", [100, 350]))
        self.line_end = tuple(lc_cfg.get("line_end", [1180, 350]))

        self.face_capture = FaceCaptureManager(output_dir=self.cfg.get("app", {}).get("captures_dir", "captures"))

        # Face Recognition: Determine active model with safe fallback to iresnet100
        rec_cfg = m_cfg.get("face_recognition", {})
        weights_path = rec_cfg.get("weights", "models/face_recognition/w600k_r50.onnx")
        if not Path(weights_path).exists():
            default_weights = "models/face_recognition/iresnet100.onnx"
            if Path(default_weights).exists():
                logger.warning(
                    f"Specified recognition model '{weights_path}' not found on disk. "
                    f"Falling back to default '{default_weights}'."
                )
                weights_path = default_weights

        model_name = Path(weights_path).stem
        emb_path = f"data/face_embeddings/registered_embeddings_{model_name}.npz"
        if not Path(emb_path).exists():
            emb_path = "data/face_embeddings/registered_embeddings.npz"

        self.extractor = EmbeddingExtractor(
            weights_path=weights_path,
            device=rec_cfg.get("device", "cpu")
        )
        self.emb_store = EmbeddingStore(storage_path=emb_path)
        self.matcher = FaceMatcher(
            threshold=rec_cfg.get("similarity_threshold", 0.40)
        )

        # Database — PostgreSQL only, config from .env
        db_cfg = self.cfg.get("database", {})
        self.db = DBConnection()
        self.entry_logger = EntryLogger(self.db)

        # Recognized track maps
        self.recognized_names: Dict[int, str] = {}
        self.recognized_scores: Dict[int, float] = {}
        self.last_seen_frame: Dict[int, int] = {}
        self.logged_tracks: Set[int] = set()
        self.frame_count: int = 0

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """Process single frame with live adaptive face re-evaluation."""
        self.frame_count += 1
        display = frame.copy()

        # 1. Person Detection (Fast YOLO imgsz=416)
        detections = self.detector.detect(frame)

        # 2. Tracking (ByteTrack)
        tracks = self.tracker.update(detections)

        # Draw virtual tripwire line only if enabled in config
        if self.line_crossing_enabled:
            LineDrawer.draw_line(display, self.line_start, self.line_end)

        active_tids: Set[int] = set()

        for tr in tracks:
            tid = tr["track_id"]
            bbox = tr["bbox"]
            active_tids.add(tid)
            self.last_seen_frame[tid] = self.frame_count

            # Extract face crop every 2 frames for active persons
            if self.frame_count % 2 == 0 or tid not in self.recognized_names:
                person_crop = safe_crop(frame, bbox)
                if person_crop is not None:
                    self.face_capture.process_person_crop(tid, person_crop, frame_num=self.frame_count)

            # Live Recognition Check:
            # Run immediately on new track, and re-evaluate every 8 frames (~0.25 sec)
            # This ensures if Harsh steps out and Kamal steps in, identity updates instantly!
            need_recognition = (tid not in self.recognized_names) or (self.frame_count % 8 == 0)

            if need_recognition:
                current_face = self.face_capture.get_best_face(tid)
                if current_face is not None:
                    emb = self.extractor.extract(current_face)
                    name, score = self.matcher.match(emb, self.emb_store.embeddings)
                    self.recognized_names[tid] = name
                    self.recognized_scores[tid] = score

            # 3. Entry logging (triggered on line crossing if enabled, or once upon face recognition)
            if self.line_crossing_enabled:
                history = self.tracker.tracks[tid]["history"]
                crossing_event = self.crossing_detector.check_crossing(tid, history)
                should_log = bool(crossing_event)
                direction = crossing_event or "in"
            else:
                current_name = self.recognized_names.get(tid)
                should_log = (tid not in self.logged_tracks and current_name is not None and current_name != "Unknown")
                direction = "in"

            if should_log:
                name = self.recognized_names.get(tid, "Unknown")
                score = self.recognized_scores.get(tid, 0.0)
                self.logged_tracks.add(tid)

                # Save snapshot
                snap_path = f"captures/person_crops/track_{tid}_{name}.jpg"
                person_crop = safe_crop(frame, bbox)
                if person_crop is not None:
                    cv2.imwrite(snap_path, person_crop)

                # Log into database
                self.entry_logger.log_entry(
                    track_id=tid,
                    person_name=name,
                    confidence=score,
                    direction=direction,
                    snapshot_path=snap_path,
                )

            # Labeling and color indicators
            label_name = self.recognized_names.get(tid, None)
            score = self.recognized_scores.get(tid, None)

            if label_name is not None and label_name != "Unknown":
                # Recognized Person: Green box
                display_label = f"{label_name} ({score:.2f})"
                box_color = (0, 255, 0)
            elif label_name == "Unknown":
                # Detected Face but Unregistered: Orange box
                display_label = "Unknown"
                box_color = (0, 140, 255)
            else:
                # Searching Face: Cyan tracking ID
                display_label = f"ID:{tid}"
                box_color = (255, 200, 0)

            draw_bounding_box(display, bbox, label=display_label, color=box_color)

        # Cleanup stale tracks (older than 30 frames / 1 sec)
        # Prevents new people from inheriting previous person's name!
        stale_tids = [
            tid for tid, last_f in self.last_seen_frame.items()
            if (tid not in active_tids and self.frame_count - last_f > 30)
        ]
        for tid in stale_tids:
            self.recognized_names.pop(tid, None)
            self.recognized_scores.pop(tid, None)
            self.last_seen_frame.pop(tid, None)
            self.logged_tracks.discard(tid)

        self.face_capture.clear_inactive_tracks(active_tids, self.frame_count, max_idle=30)

        return display
