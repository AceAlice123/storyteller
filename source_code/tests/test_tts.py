"""
Storyteller - Unit Tests for Module 2 (Text-to-Speech Synthesizer)
Tests Strategy Pattern implementations, TTSEngineFactory, WAV duration accuracy,
and database persistence to audio_assets table.
"""

import tempfile
from pathlib import Path
import pytest

from modules.tts.factory import TTSEngineFactory
from modules.tts.dummy_engine import DummyTTSEngine
from modules.tts.pyttsx3_engine import Pyttsx3Engine
from modules.tts.synthesizer import AudioSynthesizer
from database.repository import StorytellerRepository
from utils.helpers import get_audio_duration_seconds


def test_tts_engine_factory():
    """Verify that factory creates valid engine strategies and rejects invalid names."""
    pyttsx3_inst = TTSEngineFactory.create_engine("pyttsx3")
    assert isinstance(pyttsx3_inst, Pyttsx3Engine)

    dummy_inst = TTSEngineFactory.create_engine("dummy")
    assert isinstance(dummy_inst, DummyTTSEngine)

    with pytest.raises(ValueError, match="Unknown TTS engine"):
        TTSEngineFactory.create_engine("non_existent_engine")


def test_dummy_tts_engine_synthesis():
    """Verify mock engine creates playable WAV file with positive non-zero duration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_wav = Path(tmp_dir) / "test_dummy.wav"
        engine = DummyTTSEngine()

        duration = engine.synthesize(
            text="This is an automated test for the Storyteller speech subsystem.",
            output_path=out_wav
        )

        assert out_wav.is_file()
        assert out_wav.stat().st_size > 0
        assert duration > 1.0

        # Verify helper duration matches engine returned duration
        verified_duration = get_audio_duration_seconds(out_wav)
        assert abs(duration - verified_duration) < 0.05


def test_pyttsx3_engine_synthesis():
    """Verify offline pyttsx3 synthesizes a genuine WAV file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_wav = Path(tmp_dir) / "test_pyttsx3.wav"
        engine = Pyttsx3Engine()

        duration = engine.synthesize(
            text="Welcome to the Storyteller automated multimedia platform.",
            output_path=out_wav
        )

        assert out_wav.is_file()
        assert out_wav.stat().st_size > 0
        assert duration > 0.5


def test_audio_synthesizer_coordinator_with_database():
    """Verify end-to-end synthesis coordinator querying DB and updating audio_assets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "test.db"
        audio_dir = tmp_path / "temp_audio"

        repo = StorytellerRepository(db_path=db_path)
        project_id = repo.create_project("Audio Synthesis Test")

        sentences = [
            "Welcome to the first chapter of digital storytelling.",
            "Visual assets are matched seamlessly using neural embeddings."
        ]
        seq_ids = repo.insert_sequences(project_id, sentences)

        synthesizer = AudioSynthesizer(
            repository=repo,
            tts_engine="dummy",  # Use dummy for fast unit test
            temp_audio_dir=audio_dir
        )

        records = synthesizer.synthesize_project_sequences(project_id)
        assert len(records) == 2
        assert records[0]["duration_sec"] > 0
        assert Path(records[0]["file_path"]).exists()

        # Verify database audio_assets records directly
        with repo.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audio_assets ORDER BY seq_id ASC;")
            rows = cursor.fetchall()
            assert len(rows) == 2
            assert rows[0]["seq_id"] == seq_ids[0]
            assert rows[0]["duration_sec"] == records[0]["duration_sec"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
