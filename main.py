import argparse
import cv2
import sys
import yaml
from pathlib import Path
from src.pipeline import PersonEntryPipeline
from src.camera.camera_stream import CameraStream
from src.camera.video_reader import VideoReader
from src.recognition.embedding_extractor import EmbeddingExtractor
from src.face.face_detector import FaceDetector
from src.face.face_alignment import FaceAligner
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

    # Detect and align face using 5-point landmarks
    detector = FaceDetector()
    aligner = FaceAligner()
    faces = detector.detect_faces(img)
    if faces and faces[0].get("landmarks") is not None:
        face_img = aligner.align(img, landmarks=faces[0]["landmarks"])
        logger.info(f"Affine-aligned face with 5-point landmarks for '{name}'")
    elif faces:
        bbox = faces[0]["bbox"]
        x1, y1, x2, y2 = bbox
        face_img = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        logger.info(f"Detected face bbox {bbox} using ONNX det_10g.onnx")
    else:
        face_img = img
        logger.warning("No face detected by ONNX; using original image.")

    weights_path = "models/face_recognition/iresnet100.onnx"
    try:
        with open("config/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            weights_path = cfg.get("models", {}).get("face_recognition", {}).get("weights", weights_path)
    except Exception:
        pass
    model_name = Path(weights_path).stem
    emb_path = f"data/face_embeddings/registered_embeddings_{model_name}.npz"

    extractor = EmbeddingExtractor(weights_path=weights_path)
    emb = extractor.extract(face_img)
    store = EmbeddingStore(storage_path=emb_path)
    store.add_user(name, emb)
    logger.info(f"Face for '{name}' successfully registered in embeddings store!")

    # Save a clean copy of the reference face in data/registered_faces/
    user_id = name.strip().lower().replace(" ", "_")
    reg_faces_dir = Path("data/registered_faces")
    reg_faces_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = Path(image_path).suffix or ".jpg"
    target_img_path = reg_faces_dir / f"{user_id}{ext_suffix}"
    try:
        import shutil
        shutil.copyfile(image_path, str(target_img_path))
    except Exception:
        pass

    # Record user in PostgreSQL database
    try:
        db = DBConnection()
        conn = db.get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO registered_users (user_id, name, image_path, photo_uploaded)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE 
                SET name = EXCLUDED.name,
                    image_path = EXCLUDED.image_path,
                    photo_uploaded = EXCLUDED.photo_uploaded
            """, (user_id, name, str(target_img_path), True))
            conn.commit()
        conn.close()
        logger.info(f"User '{name}' recorded in PostgreSQL registered_users table (photo_uploaded=True).")
    except Exception as e:
        logger.warning(f"Failed to record registered user in PostgreSQL: {e}")


def remove_person(name: str):
    """Remove a user from all embedding stores (.npz) and PostgreSQL registered_users table."""
    logger.info(f"Removing user '{name}'...")
    import glob
    npz_files = glob.glob("data/face_embeddings/*.npz")
    removed_any = False
    for npz_path in npz_files:
        store = EmbeddingStore(storage_path=npz_path)
        if store.remove_user(name):
            removed_any = True
            logger.info(f"Removed '{name}' from {npz_path}")

    # Remove from PostgreSQL table
    try:
        db = DBConnection()
        conn = db.get_connection()
        with conn.cursor() as cur:
            user_id = name.strip().lower().replace(" ", "_")
            cur.execute("DELETE FROM registered_users WHERE user_id = %s OR LOWER(name) = %s", (user_id, name.lower()))
            deleted_rows = cur.rowcount
            conn.commit()
        conn.close()
        logger.info(f"Deleted {deleted_rows} record(s) from PostgreSQL 'registered_users' table.")
    except Exception as e:
        logger.warning(f"Could not delete from PostgreSQL: {e}")

    if removed_any:
        print(f"SUCCESS: User '{name}' has been completely removed from embeddings and database!")
    else:
        print(f"User '{name}' was not found in embeddings store.")


def run_pipeline(source: str, config_path: str, show: bool = True):
    """Run person entry pipeline on webcam, video file, or RTSP stream."""
    pipeline = PersonEntryPipeline(config_path)

    # Determine if source is live stream (webcam or RTSP/HTTP URL) or local video file
    win_title = "Person Entry Recognition"
    is_live = source.isdigit() or source.startswith(("http://", "https://", "rtsp://"))
    if is_live:
        cam_src = int(source) if source.isdigit() else source
        stream = CameraStream(source=cam_src).start()
        logger.info(f"Reading from live camera source: {source}...")
        try:
            if show:
                cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_title, 960, 540)
            while True:
                ret, frame = stream.read()
                if not ret or frame is None:
                    continue
                annotated = pipeline.process_frame(frame)
                if show:
                    cv2.imshow(win_title, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            stream.stop()
            cv2.destroyAllWindows()
    else:
        reader = VideoReader(source)
        logger.info(f"Reading from video file {source}...")
        try:
            if show:
                cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_title, 960, 540)
            for _, frame in reader.frames():
                annotated = pipeline.process_frame(frame)
                if show:
                    cv2.imshow(win_title, annotated)
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
    parser.add_argument("--remove-user", type=str, help="Remove person name from embeddings and database")
    parser.add_argument("--list-users", action="store_true", help="List all registered persons")
    parser.add_argument("--no-show", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--frontend", action="store_true", help="Launch web UI frontend dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port for web dashboard (default: 5000)")
    args = parser.parse_args()

    if args.list_users:
        store = EmbeddingStore(storage_path="data/face_embeddings/registered_embeddings_w600k_r50.npz")
        users = list(store.embeddings.keys())
        print(f"\nTotal Registered Users ({len(users)}):")
        for u in users:
            print(f"  - {u}")
        return

    if args.remove_user:
        remove_person(args.remove_user)
        return

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
