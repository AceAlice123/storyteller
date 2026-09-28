"""
Storyteller - Unit Tests for Database Repository & 3NF Schema
Tests referential integrity, foreign key enforcement, timeline JOIN queries,
and tensor embedding serialization/deserialization.
"""

import tempfile
import sqlite3
from pathlib import Path
import numpy as np
import pytest

from database.repository import StorytellerRepository


def test_database_initialization_and_tables():
    """Verify that all 5 tables from the schema are created properly."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        repo = StorytellerRepository(db_path=db_path)
        with repo.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row["name"] for row in cursor.fetchall()]

        assert "projects" in tables
        assert "sequences" in tables
        assert "audio_assets" in tables
        assert "image_assets" in tables
        assert "image_embeddings_cache" in tables
    finally:
        if db_path.exists():
            db_path.unlink()


def test_foreign_key_constraint_enforcement():
    """Verify that SQLite correctly rejects orphaned records without parent sequences."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        repo = StorytellerRepository(db_path=db_path)
        # Attempt to insert audio_asset referencing non-existent seq_id 99999
        with pytest.raises(sqlite3.IntegrityError):
            repo.insert_audio_asset(seq_id=99999, file_path="/fake/path.wav", duration_sec=3.5)
    finally:
        if db_path.exists():
            db_path.unlink()


def test_full_project_timeline_join():
    """Verify project creation, sequence insertion, asset linking, and timeline SQL JOIN."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        repo = StorytellerRepository(db_path=db_path)
        project_id = repo.create_project("AI Odyssey Test")
        assert project_id > 0

        sentences = [
            "In the beginning, machines could only calculate numbers.",
            "Today, neural networks understand images and spoken words."
        ]
        seq_ids = repo.insert_sequences(project_id, sentences)
        assert len(seq_ids) == 2

        # Link audio assets
        repo.insert_audio_asset(seq_ids[0], "/audio/clip1.wav", 4.25)
        repo.insert_audio_asset(seq_ids[1], "/audio/clip2.wav", 5.10)

        # Link image assets
        repo.insert_image_asset(seq_ids[0], "/images/vintage_computer.png", 0.884)
        repo.insert_image_asset(seq_ids[1], "/images/modern_ai_robot.png", 0.942)

        # Execute timeline JOIN
        timeline = repo.get_project_timeline(project_id)
        assert len(timeline) == 2
        assert timeline[0]["sequence_order"] == 1
        assert timeline[0]["script_text"] == sentences[0]
        assert timeline[0]["audio_file_path"] == "/audio/clip1.wav"
        assert timeline[0]["audio_duration"] == 4.25
        assert timeline[0]["image_file_path"] == "/images/vintage_computer.png"
        assert round(timeline[0]["clip_score"], 3) == 0.884

        assert timeline[1]["sequence_order"] == 2
        assert timeline[1]["audio_duration"] == 5.10
    finally:
        if db_path.exists():
            db_path.unlink()


def test_embedding_caching_and_restoration():
    """Verify caching of 512-dimensional float32 tensor embeddings in SQLite BLOB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        repo = StorytellerRepository(db_path=db_path)
        img_path = "/assets/photo_1.jpg"
        mtime = 1710000000.0
        size = 204850

        # Create dummy 512-dim embedding
        original_embedding = np.random.randn(512).astype(np.float32)

        # Cache misses before save
        cached = repo.get_cached_embedding(img_path, mtime, size)
        assert cached is None

        # Save to cache
        repo.save_cached_embedding(img_path, mtime, size, original_embedding)

        # Cache hit
        restored = repo.get_cached_embedding(img_path, mtime, size)
        assert restored is not None
        assert restored.shape == (512,)
        assert np.allclose(original_embedding, restored, atol=1e-6)

        # Cache invalidation on modified timestamp
        invalidated = repo.get_cached_embedding(img_path, mtime + 10.0, size)
        assert invalidated is None
    finally:
        if db_path.exists():
            db_path.unlink()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
