import cv2
import threading
import time
from typing import Optional, Tuple, Union
from src.utils.logger import setup_logger

logger = setup_logger("camera_stream")


class CameraStream:
    """Threaded camera capture to reduce latency for live RTSP / Webcam feeds."""

    def __init__(self, source: Union[int, str] = 0, width: Optional[int] = None, height: Optional[int] = None, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[cv2.Mat] = None
        self.ret: bool = False
        self.is_running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "CameraStream":
        logger.info(f"Connecting to video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if isinstance(self.source, int):
            if self.width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            raise RuntimeError(f"Cannot open video source {self.source}")

        self.ret, self.frame = self.cap.read()
        self.is_running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("Camera stream thread started successfully.")
        return self

    def _update(self):
        while self.is_running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info("Camera stream stopped.")
