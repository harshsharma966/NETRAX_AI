import sys
import types
import numpy as np
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

# Satisfy Ultralytics internal lap requirement using scipy (no external lap installation needed)
if "lap" not in sys.modules:
    try:
        import lap
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        def _lapjv(cost_matrix, extend_cost=True, cost_limit=1.0):
            r, c = linear_sum_assignment(cost_matrix)
            x = np.full(cost_matrix.shape[0], -1, dtype=int)
            y = np.full(cost_matrix.shape[1], -1, dtype=int)
            for i, j in zip(r, c):
                if cost_matrix[i, j] <= cost_limit:
                    x[i] = j
                    y[j] = i
            return 0.0, x, y

        mock = types.ModuleType("lap")
        mock.__version__ = "0.5.12"
        mock.lapjv = _lapjv
        sys.modules["lap"] = mock

from ultralytics.trackers.byte_tracker import BYTETracker
from src.utils.logger import setup_logger

logger = setup_logger("person_tracker")


class DetectionContainer:
    """Lightweight adapter conforming to Ultralytics Results for in-built BYTETracker."""

    def __init__(self, xywh=None, conf=None, cls=None):
        if xywh is None or len(xywh) == 0:
            self.xywh = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)
        else:
            self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
            self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
            self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return DetectionContainer(
            np.atleast_2d(self.xywh[index]),
            np.atleast_1d(self.conf[index]),
            np.atleast_1d(self.cls[index]),
        )


class PersonTracker:
    """Person tracker using Ultralytics in-built BYTETracker directly (no custom Kalman/Hungarian code)."""

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        max_lost_frames: Optional[int] = None,
        iou_threshold: Optional[float] = None,
    ):
        self.track_buffer = max_lost_frames if max_lost_frames is not None else track_buffer
        args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=(1.0 - iou_threshold) if iou_threshold is not None else match_thresh,
            track_buffer=self.track_buffer,
            fuse_score=True,
        )
        self.tracker = BYTETracker(args)
        self.tracks: Dict[int, Dict[str, Any]] = {}
        logger.info("In-built Ultralytics ByteTrack initialized successfully.")

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run Ultralytics in-built ByteTrack on current frame detections."""
        xywh_list, conf_list, cls_list = [], [], []

        for det in detections:
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            conf = float(det.get("confidence", 0.0))
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 0 or h <= 0 or conf < 0 or conf > 1:
                continue
            xywh_list.append([float(x1) + w / 2.0, float(y1) + h / 2.0, w, h])
            conf_list.append(conf)
            cls_list.append(0.0)

        container = DetectionContainer(xywh_list, conf_list, cls_list)
        results = self.tracker.update(container)

        active_tids = set()
        updated_tracks = []

        for item in results:
            x1, y1, x2, y2, tid, score, cls_id, idx = item[:8]
            tid = int(tid)
            bbox = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            conf = float(score)

            if tid not in self.tracks:
                self.tracks[tid] = {
                    "bbox": bbox,
                    "confidence": conf,
                    "history": [],
                    "lost": 0,
                }

            self.tracks[tid]["bbox"] = bbox
            self.tracks[tid]["confidence"] = conf
            self.tracks[tid]["lost"] = 0
            self.tracks[tid]["history"].append((cx, cy))
            self.tracks[tid]["history"] = self.tracks[tid]["history"][-100:]

            active_tids.add(tid)
            updated_tracks.append({
                "track_id": tid,
                "bbox": bbox,
                "confidence": conf,
                "centroid": (cx, cy),
            })

        for tid in list(self.tracks.keys()):
            if tid not in active_tids:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.track_buffer:
                    del self.tracks[tid]

        return updated_tracks
