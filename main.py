import argparse
import cv2
import sys
from pathlib import Path
from src.pipeline import PersonEntryPipeline
from src.camera.camera_stream import CameraStream
from src.camera.video_reader import VideoReader
from src.recognition.embedding_extractor import EmbeddingExtractor
from src.face.face_detector import FaceDetector
from src.database.embedding_store import EmbeddingStore
from src.database.db_connection import DBConnection
from src.utils.logger import setup_logger

logger = setup_logger("main")


def register_face(name: str, image_path: str):
    """Register a new face into gallery embeddings and PostgreSQL."""
    logger.info(f"Registering face for '{name}' using image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image at {image_path}")
        return

    # Detect and crop face using ONNX SCRFD FaceDetector
    detector = FaceDetector()
    faces = detector.detect_faces(img)
    if faces:
        bbox = faces[0]["bbox"]
        x1, y1, x2, y2 = bbox
        face_img = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        logger.info(f"Detected face bbox {bbox} using ONNX det_10g.onnx")
    else:
        face_img = img
        logger.warning("No face detected by ONNX; using original image.")

    extractor = EmbeddingExtractor()
    emb = extractor.extract(face_img)
    store = EmbeddingStore()
    store.add_user(name, emb)
    logger.info(f"Face for '{name}' successfully registered in embeddings store!")

    # Record user in PostgreSQL database
    try:
        db = DBConnection()
        conn = db.get_connection()
        with conn.cursor() as cur:
            user_id = name.strip().lower().replace(" ", "_")
            cur.execute("""
                INSERT INTO registered_users (user_id, name, image_path, photo_uploaded)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE 
                SET name = EXCLUDED.name,
                    image_path = EXCLUDED.image_path,
                    photo_uploaded = EXCLUDED.photo_uploaded
            """, (user_id, name, image_path, True))
            conn.commit()
        conn.close()
        logger.info(f"User '{name}' recorded in PostgreSQL registered_users table (photo_uploaded=True).")
    except Exception as e:
        logger.warning(f"Failed to record registered user in PostgreSQL: {e}")


def run_pipeline(source: str, config_path: str, show: bool = True):
    """Run person entry pipeline on webcam, video file, or RTSP stream."""
    pipeline = PersonEntryPipeline(config_path)

    # Determine if source is live stream (webcam or RTSP/HTTP URL) or local video file
    is_live = source.isdigit() or source.startswith(("http://", "https://", "rtsp://"))
    if is_live:
        cam_src = int(source) if source.isdigit() else source
        stream = CameraStream(source=cam_src).start()
        logger.info(f"Reading from live camera source: {source}...")
        try:
            while True:
                ret, frame = stream.read()
                if not ret or frame is None:
                    continue
                annotated = pipeline.process_frame(frame)
                if show:
                    cv2.imshow("Person Entry Recognition", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            stream.stop()
            cv2.destroyAllWindows()
    else:
        reader = VideoReader(source)
        logger.info(f"Reading from video file {source}...")
        try:
            for _, frame in reader.frames():
                annotated = pipeline.process_frame(frame)
                if show:
                    cv2.imshow("Person Entry Recognition", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            reader.release()
            cv2.destroyAllWindows()


def start_frontend(port: int = 5000):
    """Function to launch frontend web UI dashboard."""
    from frontend.app import run_frontend
    run_frontend(port=port)


def main():
    parser = argparse.ArgumentParser(description="Person Entry Recognition System")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0) or path to video file / RTSP stream")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--register", type=str, help="Register new person name")
    parser.add_argument("--image", type=str, help="Face image path for registration")
    parser.add_argument("--no-show", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--frontend", action="store_true", help="Launch web UI frontend dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port for web dashboard (default: 5000)")
    args = parser.parse_args()

    if args.frontend:
        start_frontend(port=args.port)
        return

    if args.register:
        if not args.image:
            logger.error("--image is required when using --register")
            sys.exit(1)
        register_face(args.register, args.image)
        return

    run_pipeline(source=args.source, config_path=args.config, show=not args.no_show)


if __name__ == "__main__":
    main()
