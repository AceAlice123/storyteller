"""
Storyteller - Unit Tests for Module 1 (Input Parser & Orchestrator)
Tests regex sanitization, abbreviation protection, sentence splitting,
file ingestion, and database sequence persistence.
"""

import tempfile
from pathlib import Path
import pytest

from modules.parser.text_parser import TextParser
from database.repository import StorytellerRepository


def test_text_cleaning():
    """Verify removal of illegal unicode, normalization of smart quotes and em-dashes."""
    parser = TextParser()
    raw_input = '“Artificial Intelligence,” Dr. Alan said—it’s the future! 🚀\t\r\nHere is   more text.'
    cleaned = parser.clean_text(raw_input)

    assert "“" not in cleaned
    assert "”" not in cleaned
    assert '"' in cleaned
    assert "🚀" not in cleaned  # Emoji stripped
    assert "  " not in cleaned  # Double space collapsed
    assert "Artificial Intelligence" in cleaned


def test_sentence_splitting_with_abbreviations():
    """Verify sentence splitting does not prematurely split on titles or common abbreviations."""
    parser = TextParser()
    text = "Dr. Alan Turing was an English mathematician. He developed the Bombe machine in 1940. What an achievement! Isn't that incredible?"
    sentences = parser.split_into_sentences(text)

    assert len(sentences) == 4
    assert sentences[0] == "Dr. Alan Turing was an English mathematician."
    assert sentences[1] == "He developed the Bombe machine in 1940."
    assert sentences[2] == "What an achievement!"
    assert sentences[3] == "Isn't that incredible?"


def test_parse_file_and_ingest():
    """Verify end-to-end file reading, project creation, and sequence persistence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        script_file = tmp_path / "script.txt"
        db_file = tmp_path / "test.db"

        script_content = (
            "Deep learning has transformed computer vision.\n\n"
            "Pretrained models can map visual semantics into shared vector spaces. "
            "This empowers zero-shot multimodal retrieval."
        )
        script_file.write_text(script_content, encoding="utf-8")

        repo = StorytellerRepository(db_path=db_file)
        parser = TextParser()

        project_id, seq_ids, sentences = parser.ingest_script(
            script_path=script_file,
            project_name="Deep Learning Video",
            repository=repo
        )

        assert project_id == 1
        assert len(seq_ids) == 3
        assert len(sentences) == 3

        # Verify DB records
        db_sequences = repo.get_sequences(project_id)
        assert len(db_sequences) == 3
        assert db_sequences[0]["sequence_order"] == 1
        assert db_sequences[0]["script_text"] == "Deep learning has transformed computer vision."
        assert db_sequences[1]["sequence_order"] == 2
        assert db_sequences[2]["sequence_order"] == 3


def test_empty_file_handling():
    """Verify graceful error handling on empty or whitespace-only files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("   \n\n  \t  ", encoding="utf-8")

        parser = TextParser()
        with pytest.raises(ValueError, match="empty or contains only whitespace"):
            parser.parse_file(empty_file)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
