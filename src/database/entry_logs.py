import os
from typing import Optional
from pathlib import Path
from src.database.db_connection import DBConnection
from src.utils.logger import setup_logger

logger = setup_logger("entry_logs")


class EntryLogger:
    """Logs confirmed crossing and recognition events into PostgreSQL database."""

    def __init__(self, db_conn: DBConnection):
        self.db = db_conn
        self.photo_upload_dir = Path(
            os.getenv("PHOTO_UPLOAD_DIR", "captures/uploads")
        )
        self.photo_upload_dir.mkdir(parents=True, exist_ok=True)

    def log_entry(
        self,
        track_id: int,
        person_name: str,
        confidence: float,
        direction: str,
        snapshot_path: Optional[str] = None,
        photo_path: Optional[str] = None,
    ):
        """Log a crossing event. Optionally attach a photo upload path."""
        photo_uploaded = photo_path is not None and Path(photo_path).exists()
        placeholder = self.db.placeholder

        query = f"""
            INSERT INTO entry_logs
                (track_id, person_name, confidence, direction,
                 snapshot_path, photo_path, photo_uploaded)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder},
                    {placeholder}, {placeholder}, {placeholder})
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    track_id,
                    person_name,
                    confidence,
                    direction,
                    snapshot_path or "",
                    photo_path or "",
                    photo_uploaded,
                ),
            )
            conn.commit()
            conn.close()
            uploaded_status = "✓ photo uploaded" if photo_uploaded else "✗ no photo"
            logger.info(
                f"Logged entry: [Track {track_id}] {person_name} "
                f"({direction}) conf={confidence:.2f} | {uploaded_status}"
            )
        except Exception as e:
            logger.error(f"Failed to log entry into PostgreSQL: {e}")

    def check_photo_uploaded(self, entry_id: int) -> bool:
        """Check if a photo has been uploaded for a given entry log ID."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT photo_uploaded FROM entry_logs WHERE id = %s", (entry_id,)
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return bool(row[0])
            return False
        except Exception as e:
            logger.error(f"Failed to check photo upload status: {e}")
            return False

    def update_photo(self, entry_id: int, photo_path: str):
        """Update an existing entry log with a photo path after upload."""
        photo_exists = Path(photo_path).exists()
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE entry_logs
                SET photo_path = %s, photo_uploaded = %s
                WHERE id = %s
                """,
                (photo_path, photo_exists, entry_id),
            )
            conn.commit()
            conn.close()
            logger.info(
                f"Updated entry {entry_id} with photo: {photo_path} "
                f"(exists={photo_exists})"
            )
        except Exception as e:
            logger.error(f"Failed to update photo for entry {entry_id}: {e}")
