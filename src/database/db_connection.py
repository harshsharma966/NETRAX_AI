import os
from typing import Optional
from dotenv import load_dotenv
from src.utils.logger import setup_logger

# Load .env file at module level
load_dotenv()

logger = setup_logger("db_connection")


class DBConnection:
    """Database connection manager — PostgreSQL only (reads config from .env)."""

    def __init__(self):
        self.placeholder = "%s"
        self.pg_host = os.getenv("POSTGRES_HOST", "db")
        self.pg_port = int(os.getenv("POSTGRES_PORT", 5432))
        self.pg_db = os.getenv("POSTGRES_DB", "netrax_db")
        self.pg_user = os.getenv("POSTGRES_USER", "postgres")
        self.pg_password = os.getenv("POSTGRES_PASSWORD", "password123")

        self._init_tables()

    def get_connection(self):
        """Return a live PostgreSQL database connection."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL. "
                "Install it using: pip install psycopg2-binary"
            )
        return psycopg2.connect(
            host=self.pg_host,
            port=self.pg_port,
            dbname=self.pg_db,
            user=self.pg_user,
            password=self.pg_password,
        )

    def _init_tables(self):
        """Initialize database schema in PostgreSQL."""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS registered_users (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100) UNIQUE NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        image_path TEXT,
                        photo_uploaded BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entry_logs (
                        id SERIAL PRIMARY KEY,
                        track_id INT NOT NULL,
                        person_name VARCHAR(255) NOT NULL,
                        confidence REAL NOT NULL,
                        direction VARCHAR(50) NOT NULL,
                        snapshot_path TEXT,
                        photo_path TEXT,
                        photo_uploaded BOOLEAN DEFAULT FALSE,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
            conn.close()
            logger.info(
                f"PostgreSQL Database '{self.pg_db}' at "
                f"{self.pg_host}:{self.pg_port} initialized successfully."
            )
        except Exception as e:
            logger.warning(
                f"PostgreSQL connection to '{self.pg_host}:{self.pg_port}/{self.pg_db}' "
                f"failed ({e}). Make sure PostgreSQL is running and .env credentials are correct."
            )
