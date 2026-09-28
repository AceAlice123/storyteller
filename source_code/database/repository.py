"""
Storyteller - Automated Text-to-Video Synthesis Platform
Database Repository Layer (Repository Pattern)

Encapsulates all persistence logic, schema migrations, and queries for SQLite.
Enforces 3NF referential integrity, WAL mode, foreign keys, and parameterized queries
to protect against SQL injection vulnerabilities.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
import contextlib
import numpy as np

from config import DATABASE_PATH, SCHEMA_PATH
from utils.logger import setup_logger

logger = setup_logger("Storyteller.Repository")


class StorytellerRepository:
    """
    Repository pattern implementation for SQLite database operations.
    Provides decoupled data access methods for projects, sequences, audio, images,
    and tensor embedding caching.
    """

    def __init__(self, db_path: Path | str = DATABASE_PATH):
        self.db_path = Path(db_path).resolve()
        self._ensure_parent_directory()
        self.initialize_schema()

    def _ensure_parent_directory(self) -> None:
        """Ensures the parent directory for the SQLite database exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def get_connection(self):
        """
        Creates, configures, and safely yields an SQLite connection, ensuring it is closed.
        Enables foreign keys and WAL mode.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def initialize_schema(self, schema_file: Path | str = SCHEMA_PATH) -> None:
        """
        Executes the DDL schema file to create required tables and indices if not already present.
        """
        schema_path = Path(schema_file).resolve()
        if not schema_path.is_file():
            raise FileNotFoundError(f"Schema file not found at: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            ddl_script = f.read()

        with self.get_connection() as conn:
            conn.executescript(ddl_script)
            conn.commit()
        logger.debug(f"Database schema verified/initialized at: {self.db_path}")

    # -------------------------------------------------------------------------
    # Project Operations
    # -------------------------------------------------------------------------
    def create_project(self, project_name: str) -> int:
        """
        Inserts a new project record and returns the generated project_id.

        Args:
            project_name: Descriptive name for the video synthesis run.

        Returns:
            int: The unique project_id.
        """
        query = "INSERT INTO projects (project_name) VALUES (?);"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (project_name.strip(),))
            conn.commit()
            project_id = cursor.lastrowid
            logger.info(f"Created Project (ID: {project_id}) - '{project_name}'")
            return project_id

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single project record by its primary key."""
        query = "SELECT project_id, project_name, creation_timestamp FROM projects WHERE project_id = ?;"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (project_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------------------
    # Sequence Operations
    # -------------------------------------------------------------------------
    def insert_sequences(self, project_id: int, text_segments: List[str]) -> List[int]:
        """
        Inserts a batch of sanitized script sentences into the sequences table.

        Args:
            project_id: Foreign key reference to projects.project_id.
            text_segments: Ordered list of sanitized text sentences.

        Returns:
            List[int]: List of inserted seq_id primary keys in sequential order.
        """
        query = """
            INSERT INTO sequences (project_id, sequence_order, script_text)
            VALUES (?, ?, ?);
        """
        seq_ids: List[int] = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for idx, text in enumerate(text_segments, start=1):
                cursor.execute(query, (project_id, idx, text))
                seq_ids.append(cursor.lastrowid)
            conn.commit()
        logger.info(f"Inserted {len(seq_ids)} sequences for Project ID: {project_id}.")
        return seq_ids

    def get_sequences(self, project_id: int) -> List[Dict[str, Any]]:
        """Retrieves all sequence records for a given project ordered chronologically."""
        query = """
            SELECT seq_id, project_id, sequence_order, script_text
            FROM sequences
            WHERE project_id = ?
            ORDER BY sequence_order ASC;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (project_id,))
            return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Audio Asset Operations
    # -------------------------------------------------------------------------
    def insert_audio_asset(self, seq_id: int, file_path: str, duration_sec: float) -> int:
        """
        Records synthesized audio clip metadata for a sequence.

        Args:
            seq_id: Foreign key reference to sequences.seq_id.
            file_path: Absolute filesystem path to generated .wav file.
            duration_sec: Exact duration in seconds.

        Returns:
            int: audio_id primary key.
        """
        query = """
            INSERT INTO audio_assets (seq_id, file_path, duration_sec)
            VALUES (?, ?, ?)
            ON CONFLICT(seq_id) DO UPDATE SET
                file_path = excluded.file_path,
                duration_sec = excluded.duration_sec;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (seq_id, str(file_path), float(duration_sec)))
            conn.commit()
            return cursor.lastrowid

    # -------------------------------------------------------------------------
    # Image Asset Operations
    # -------------------------------------------------------------------------
    def insert_image_asset(self, seq_id: int, file_path: str, clip_score: float) -> int:
        """
        Records the AI-matched visual asset for a sequence.

        Args:
            seq_id: Foreign key reference to sequences.seq_id.
            file_path: Absolute filesystem path to selected image.
            clip_score: Cosine similarity / softmax confidence score.

        Returns:
            int: image_id primary key.
        """
        query = """
            INSERT INTO image_assets (seq_id, file_path, clip_score)
            VALUES (?, ?, ?)
            ON CONFLICT(seq_id) DO UPDATE SET
                file_path = excluded.file_path,
                clip_score = excluded.clip_score;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (seq_id, str(file_path), float(clip_score)))
            conn.commit()
            return cursor.lastrowid

    # -------------------------------------------------------------------------
    # Timeline Query (SQL JOIN) for Video Compositor
    # -------------------------------------------------------------------------
    def get_project_timeline(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Executes a 3NF SQL JOIN across sequences, audio_assets, and image_assets.
        Fetches the complete chronological timeline required by the MoviePy compositor.

        Returns:
            List[Dict[str, Any]]: Ordered sequence dictionaries with audio and image paths.
        """
        query = """
            SELECT 
                s.seq_id,
                s.project_id,
                s.sequence_order,
                s.script_text,
                a.audio_id,
                a.file_path AS audio_file_path,
                a.duration_sec AS audio_duration,
                i.image_id,
                i.file_path AS image_file_path,
                i.clip_score
            FROM sequences s
            JOIN audio_assets a ON s.seq_id = a.seq_id
            JOIN image_assets i ON s.seq_id = i.seq_id
            WHERE s.project_id = ?
            ORDER BY s.sequence_order ASC;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (project_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # -------------------------------------------------------------------------
    # CLIP Image Embedding Cache Operations
    # -------------------------------------------------------------------------
    def get_cached_embedding(
        self, image_path: str, file_mtime: float, file_size: int
    ) -> Optional[np.ndarray]:
        """
        Retrieves a cached 512-d CLIP embedding for an image if modification time and size match.
        Prevents redundant heavy neural network inference.
        """
        query = """
            SELECT embedding_blob 
            FROM image_embeddings_cache 
            WHERE image_path = ? AND file_mtime = ? AND file_size = ?;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (str(image_path), float(file_mtime), int(file_size)))
            row = cursor.fetchone()
            if row and row["embedding_blob"]:
                # Convert raw byte blob back to 1D float32 numpy array
                return np.frombuffer(row["embedding_blob"], dtype=np.float32)
        return None

    def save_cached_embedding(
        self, image_path: str, file_mtime: float, file_size: int, embedding: np.ndarray
    ) -> None:
        """
        Caches a 512-d numpy embedding vector as a raw BLOB in SQLite.
        """
        blob = embedding.astype(np.float32).tobytes()
        query = """
            INSERT INTO image_embeddings_cache (image_path, file_mtime, file_size, embedding_blob)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(image_path) DO UPDATE SET
                file_mtime = excluded.file_mtime,
                file_size = excluded.file_size,
                embedding_blob = excluded.embedding_blob,
                updated_at = CURRENT_TIMESTAMP;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (str(image_path), float(file_mtime), int(file_size), blob))
            conn.commit()

    def clear_embedding_cache(self) -> int:
        """Purges all entries from the image embedding cache."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM image_embeddings_cache;")
            count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleared {count} cached embeddings.")
            return count
