import cv2
from typing import Tuple, Optional, Generator
from pathlib import Path


class VideoReader:
    """Sequential video file reader for processing recorded video files."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Generator[Tuple[int, cv2.Mat], None, None]:
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
