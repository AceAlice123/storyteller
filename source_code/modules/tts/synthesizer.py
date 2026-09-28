"""
Storyteller - Automated Text-to-Video Synthesis Platform
Module 2: Text-to-Speech (TTS) Synthesizer Coordinator
Course: IGNOU MCA MCSP-232

Responsible for:
1. Fetching ordered sequence records from SQLite for the active project.
2. Generating discrete .wav audio files using the selected BaseTTSEngine strategy.
3. Calculating exact mathematical floating-point duration for each audio segment.
4. Persisting file paths and duration metrics to the audio_assets database table.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import os

from database.repository import StorytellerRepository
from modules.tts.base import BaseTTSEngine
from modules.tts.factory import TTSEngineFactory
from config import DEFAULT_TEMP_AUDIO_DIR
from utils.logger import setup_logger

logger = setup_logger("Storyteller.Synthesizer")


class AudioSynthesizer:
    """
    Coordinates speech synthesis across all sequences of a project.
    Bridges the text sequences from the database with the TTS Strategy Engine
    and persists generated audio metadata.
    """

    def __init__(
        self,
        repository: StorytellerRepository,
        tts_engine: BaseTTSEngine | str = "pyttsx3",
        temp_audio_dir: Path | str = DEFAULT_TEMP_AUDIO_DIR
    ):
        self.repository = repository
        self.temp_audio_dir = Path(temp_audio_dir).resolve()
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(tts_engine, str):
            self.engine = TTSEngineFactory.create_engine(tts_engine)
        else:
            self.engine = tts_engine

        logger.info(f"AudioSynthesizer initialized using engine: {self.engine.engine_name}")

    def synthesize_project_sequences(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Processes all sequences for a project, synthesizes audio files in batch,
        and logs them to the audio_assets table.

        Args:
            project_id: Primary key of the project in projects table.

        Returns:
            List[Dict[str, Any]]: List of created audio asset records with durations.

        Raises:
            ValueError: If no sequences are found for the project.
        """
        sequences = self.repository.get_sequences(project_id)
        if not sequences:
            raise ValueError(f"No sequences found in database for Project ID: {project_id}")

        logger.info(
            f"Synthesizing audio for {len(sequences)} sequences (Project ID: {project_id})..."
        )

        # Prepare batch items for TTS Strategy
        batch_items: List[Tuple[str, Path| str]] = []
        for seq in sequences:
            seq_order = seq["sequence_order"]
            text = seq["script_text"]
            audio_filename = f"proj_{project_id}_seq_{seq_order:03d}.wav"
            audio_path = self.temp_audio_dir / audio_filename
            batch_items.append((text, audio_path))

        # Perform synthesis via BaseTTSEngine strategy
        durations = self.engine.synthesize_batch(batch_items)

        audio_records: List[Dict[str, Any]] = []
        for seq, (text, audio_path), duration in zip(sequences, batch_items, durations):
            seq_id = seq["seq_id"]
            seq_order = seq["sequence_order"]

            # Persist to database audio_assets table
            audio_id = self.repository.insert_audio_asset(
                seq_id=seq_id,
                file_path=str(audio_path),
                duration_sec=duration
            )

            record = {
                "audio_id": audio_id,
                "seq_id": seq_id,
                "sequence_order": seq_order,
                "file_path": str(audio_path),
                "duration_sec": duration,
                "script_text": text
            }
            audio_records.append(record)

        total_duration = sum(r["duration_sec"] for r in audio_records)
        logger.info(
            f"Successfully synthesized {len(audio_records)} audio assets. "
            f"Total narration runtime: {total_duration:.2f} seconds."
        )
        return audio_records
