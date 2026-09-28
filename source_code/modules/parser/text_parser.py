"""
Storyteller - Automated Text-to-Video Synthesis Platform
Module 1: Input Parser and Orchestrator
Course: IGNOU MCA MCSP-232

Responsible for:
1. Safe file ingestion with UTF-8 decoding.
2. Regex-based string sanitization to remove illegal symbols that crash offline TTS engines.
3. Sentence/block segmentation based on terminal punctuation and line boundaries.
4. Database state instantiation: creating the project record and ordered sequence rows.
"""

from pathlib import Path
from typing import List, Tuple
import re

from database.repository import StorytellerRepository
from utils.logger import setup_logger

logger = setup_logger("Storyteller.TextParser")


class TextParser:
    """
    Parses and sanitizes narrative scripts into discrete, well-formed sentence blocks.
    Acts as the entrypoint for raw textual content into the Storyteller pipeline.
    """

    # Common abbreviations to prevent premature sentence splitting
    ABBREVIATIONS = (
        r"\b(mr|mrs|ms|dr|prof|sr|jr|vs|etc|fig|e\.g|i\.e)\.",
    )

    def __init__(self):
        # Precompile regex for performance
        self._whitespace_pattern = re.compile(r"\s+")
        # Match terminal punctuation followed by space or newline, while avoiding common abbreviations
        self._sentence_splitter = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

    def clean_text(self, raw_text: str) -> str:
        """
        Sanitizes raw text:
        - Normalizes smart quotes and typographic dashes to ASCII equivalents.
        - Strips control characters and non-standard symbols that crash offline TTS engines.
        - Normalizes excessive whitespace.

        Args:
            raw_text: Uncleaned input string.

        Returns:
            str: Normalized, TTS-safe string.
        """
        if not raw_text:
            return ""

        # Normalize typographic quotes and dashes
        replacements = {
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "—": " - ", "–": " - ", "…": "...",
            "\r\n": "\n", "\r": "\n"
        }
        text = raw_text
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Retain standard alphanumerics, basic punctuation, and whitespace
        # Strips emojis, control chars, and exotic unicode symbols
        text = re.sub(r"[^\w\s.,!?'\":;\-–—\(\)]", " ", text, flags=re.UNICODE)

        # Collapse repeated spaces/tabs (preserve single newlines for paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def split_into_sentences(self, cleaned_text: str) -> List[str]:
        """
        Splits cleaned text into discrete sentence blocks using punctuation markers
        and paragraph breaks.

        Args:
            cleaned_text: Sanitized string.

        Returns:
            List[str]: Array of sentence strings ready for audio synthesis and CLIP matching.
        """
        if not cleaned_text:
            return []

        # First split by paragraph / double newline
        paragraphs = [p.strip() for p in cleaned_text.split("\n") if p.strip()]
        sentences: List[str] = []

        for para in paragraphs:
            # Protect abbreviations by temporarily replacing their dot
            protected_para = para
            for abbr in self.ABBREVIATIONS:
                protected_para = re.sub(
                    abbr,
                    lambda m: m.group(0).replace(".", "@@DOT@@"),
                    protected_para,
                    flags=re.IGNORECASE
                )

            # Split on terminal punctuation followed by whitespace or uppercase start
            raw_splits = re.split(r"(?<=[.!?])\s+", protected_para)

            for segment in raw_splits:
                # Restore protected dots
                restored = segment.replace("@@DOT@@", ".").strip()
                if restored:
                    # Ensure minimal length for meaningful narration
                    if len(restored) > 2:
                        sentences.append(restored)

        logger.debug(f"Segmented text into {len(sentences)} distinct sequences.")
        return sentences

    def parse_file(self, script_path: Path | str) -> List[str]:
        """
        Reads, sanitizes, and segments a plain .txt script file.

        Args:
            script_path: Path to script file.

        Returns:
            List[str]: Cleaned list of sentence strings.

        Raises:
            FileNotFoundError: If the script file does not exist.
            ValueError: If the script is empty.
        """
        path = Path(script_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Script file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback for systems saving in Latin-1 / Windows-1252
            logger.warning(f"UTF-8 decoding failed for {path}; falling back to Latin-1.")
            with open(path, "r", encoding="latin-1") as f:
                content = f.read()

        cleaned = self.clean_text(content)
        if not cleaned.strip():
            raise ValueError(f"Script file '{path}' is empty or contains only whitespace.")

        sentences = self.split_into_sentences(cleaned)
        if not sentences:
            raise ValueError(f"No valid narrative sentences could be extracted from '{path}'.")

        logger.info(f"Successfully parsed {len(sentences)} sentences from '{path.name}'.")
        return sentences

    def ingest_script(
        self,
        script_path: Path | str,
        project_name: str,
        repository: StorytellerRepository
    ) -> Tuple[int, List[int], List[str]]:
        """
        High-level orchestrator method for Module 1.
        Parses script, creates the project record, and inserts the sequences into SQLite.

        Args:
            script_path: Path to input script.
            project_name: Name of the video synthesis project.
            repository: StorytellerRepository instance.

        Returns:
            Tuple[int, List[int], List[str]]: (project_id, list of seq_ids, list of sentences)
        """
        sentences = self.parse_file(script_path)
        project_id = repository.create_project(project_name)
        seq_ids = repository.insert_sequences(project_id, sentences)
        return project_id, seq_ids, sentences
