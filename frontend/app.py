import os
import cv2
import time
import threading
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

from src.database.db_connection import DBConnection
from src.database.embedding_store import EmbeddingStore
from src.recognition.embedding_extractor import EmbeddingExtractor
from src.pipeline import PersonEntryPipeline
from src.utils.logger import setup_logger

logger = setup_logger("frontend_app")

# Shared State
_pipeline = None
_extractor = None
_emb_store = None
_pipeline_lock = threading.Lock()
_camera_running = False
_camera_source = os.getenv("DEFAULT_CAMERA_SOURCE", "0")
_video_cap = None


def get_pipeline():
    global _pipeline, _extractor, _emb_store
    if _pipeline is None:
        try:
            logger.info("Initializing PersonEntryPipeline for frontend...")
            _pipeline = PersonEntryPipeline("config/config.yaml")
            _extractor = EmbeddingExtractor()
            _emb_store = EmbeddingStore()
            logger.info("Pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
    return _pipeline


def generate_frames():
    """MJPEG stream generator from camera/video."""
    global _video_cap, _camera_running, _camera_source
    pipe = get_pipeline()

    if _video_cap is None or not _video_cap.isOpened():
        source = int(_camera_source) if str(_camera_source).isdigit() else _camera_source
        _video_cap = cv2.VideoCapture(source)
        try:
            _video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    while _camera_running:
        if _video_cap is None or not _video_cap.isOpened():
            time.sleep(0.5)
            continue

        ret, frame = _video_cap.read()
        if not ret or frame is None:
            if not str(_camera_source).isdigit():
                _video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.05)
            continue

        with _pipeline_lock:
            if pipe is not None:
                try:
                    annotated = pipe.process_frame(frame)
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    annotated = frame
            else:
                annotated = frame

        ret, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)


def create_app(base_dir=None):
    """Factory function that creates and configures the Flask Frontend Application."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    templates_dir = Path(__file__).resolve().parent / "templates"
    static_dir = Path(__file__).resolve().parent / "static"
    reg_faces_dir = base_dir / "data" / "registered_faces"
    captures_dir = base_dir / "captures"

    reg_faces_dir.mkdir(parents=True, exist_ok=True)
    captures_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=str(templates_dir), static_folder=str(static_dir))

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        global _camera_running
        _camera_running = True
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/camera/toggle", methods=["POST"])
    def toggle_camera():
        global _camera_running, _video_cap
        data = request.get_json(silent=True) or {}
        action = data.get("action", "toggle")

        if action == "start":
            _camera_running = True
        elif action == "stop":
            _camera_running = False
            if _video_cap is not None:
                _video_cap.release()
                _video_cap = None
        else:
            _camera_running = not _camera_running
            if not _camera_running and _video_cap is not None:
                _video_cap.release()
                _video_cap = None

        return jsonify({"camera_running": _camera_running})

    @app.route("/api/register", methods=["POST"])
    def register():
        """Register person with Name + Photo upload."""
        try:
            name = request.form.get("name", "").strip()
            if not name:
                return jsonify({"success": False, "error": "Person name is required"}), 400

            if "photo" not in request.files or request.files["photo"].filename == "":
                return jsonify({"success": False, "error": "Photo image file is required"}), 400

            photo_file = request.files["photo"]
            filename = f"{name.lower().replace(' ', '_')}_{int(time.time())}.jpg"
            save_path = reg_faces_dir / filename
            photo_file.save(str(save_path))

            img = cv2.imread(str(save_path))
            if img is None:
                return jsonify({"success": False, "error": "Invalid image file uploaded"}), 400

            # Detect and crop face using ONNX det_10g.onnx
            from src.face.face_detector import FaceDetector
            detector = FaceDetector()
            faces = detector.detect_faces(img)
            if faces:
                bbox = faces[0]["bbox"]
                x1, y1, x2, y2 = bbox
                face_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            else:
                face_crop = img

            if _pipeline is not None and hasattr(_pipeline, "extractor") and hasattr(_pipeline, "emb_store"):
                ext = _pipeline.extractor
                store = _pipeline.emb_store
            else:
                ext = EmbeddingExtractor()
                store = EmbeddingStore()
            emb = ext.extract(face_crop)
            store.add_user(name, emb)

            if _pipeline is not None and hasattr(_pipeline, "emb_store"):
                _pipeline.emb_store.load()

            # Record in PostgreSQL
            db = DBConnection()
            conn = db.get_connection()
            user_id = name.lower().replace(" ", "_")
            rel_img_path = f"data/registered_faces/{filename}"
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO registered_users (user_id, name, image_path, photo_uploaded)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE
                    SET name = EXCLUDED.name,
                        image_path = EXCLUDED.image_path,
                        photo_uploaded = EXCLUDED.photo_uploaded
                """, (user_id, name, rel_img_path, True))
                conn.commit()
            conn.close()

            logger.info(f"Registered user '{name}' via frontend UI.")
            return jsonify({
                "success": True,
                "message": f"Successfully registered '{name}' with photo!",
                "user": {"name": name, "user_id": user_id, "image_path": rel_img_path}
            })
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/logs")
    def get_logs():
        try:
            db = DBConnection()
            conn = db.get_connection()
            logs = []
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, track_id, person_name, confidence, direction, snapshot_path, photo_uploaded, timestamp
                    FROM entry_logs
                    ORDER BY timestamp DESC
                    LIMIT 50;
                """)
                rows = cur.fetchall()
                for r in rows:
                    logs.append({
                        "id": r[0],
                        "track_id": r[1],
                        "name": r[2],
                        "confidence": round(float(r[3]), 2),
                        "direction": r[4],
                        "snapshot_path": r[5] or "",
                        "photo_uploaded": bool(r[6]),
                        "timestamp": r[7].strftime("%Y-%m-%d %H:%M:%S") if r[7] else ""
                    })
            conn.close()
            return jsonify({"success": True, "logs": logs})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/users")
    def get_users():
        try:
            db = DBConnection()
            conn = db.get_connection()
            users = []
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, name, image_path, photo_uploaded, created_at
                    FROM registered_users
                    ORDER BY created_at DESC;
                """)
                rows = cur.fetchall()
                for r in rows:
                    users.append({
                        "id": r[0],
                        "user_id": r[1],
                        "name": r[2],
                        "image_path": r[3] or "",
                        "photo_uploaded": bool(r[4]),
                        "created_at": r[5].strftime("%Y-%m-%d %H:%M:%S") if r[5] else ""
                    })
            conn.close()
            return jsonify({"success": True, "users": users})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/stats")
    def get_stats():
        try:
            db = DBConnection()
            conn = db.get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM registered_users;")
                total_users = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM entry_logs;")
                total_entries = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM entry_logs WHERE person_name != 'Unknown';")
                recognized_entries = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM entry_logs WHERE person_name = 'Unknown';")
                unknown_entries = cur.fetchone()[0]

            conn.close()
            return jsonify({
                "success": True,
                "total_users": total_users,
                "total_entries": total_entries,
                "recognized_entries": recognized_entries,
                "unknown_entries": unknown_entries,
                "camera_running": _camera_running
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/captures/<path:filename>")
    def serve_capture(filename):
        return send_from_directory(str(captures_dir), filename)

    @app.route("/data/registered_faces/<path:filename>")
    def serve_registered_face(filename):
        return send_from_directory(str(reg_faces_dir), filename)

    @app.route("/image/<path:filepath>")
    def serve_any_image(filepath):
        full_path = base_dir / filepath
        if full_path.exists():
            return send_from_directory(str(full_path.parent), full_path.name)
        return "", 404

    return app


def run_frontend(host="0.0.0.0", port=5000):
    """Function to launch and run the frontend web dashboard."""
    port = int(os.getenv("PORT", port))
    print(f"\n" + "="*56)
    print(f"  NETRAX AI Frontend Dashboard: http://localhost:{port}")
    print(f"="*56 + "\n")
    application = create_app()
    application.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_frontend()

