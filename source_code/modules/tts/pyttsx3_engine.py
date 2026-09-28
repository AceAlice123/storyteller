"""
Storyteller - Automated Text-to-Video Synthesis Platform
Concrete TTS Strategy: Offline pyttsx3 Engine

Uses the local pyttsx3 multi-platform library (Windows SAPI5 / Linux eSpeak)
to produce offline, zero-cost audio narration files.
"""

from pathlib import Path
from typing import List, Tuple
import os
import pyttsx3

from modules.tts.base import BaseTTSEngine
from utils.helpers import get_audio_duration_seconds
from utils.logger import setup_logger
from config import TTS_SPEECH_RATE, TTS_VOLUME

logger = setup_logger("Storyteller.Pyttsx3Engine")


class Pyttsx3Engine(BaseTTSEngine):
    """
    Concrete TTS Strategy utilizing pyttsx3 for completely offline speech generation.
    Queues batch items to prevent Windows SAPI event loop lockups.
    """

    def __init__(self, rate: int = TTS_SPEECH_RATE, volume: float = TTS_VOLUME):
        self.rate = rate
        self.volume = volume

    @property
    def engine_name(self) -> str:
        return "pyttsx3 (Offline Local)"

    def synthesize_batch(self, items: List[Tuple[str, Path | str]]) -> List[float]:
        """
        Synthesizes a list of (text, file_path) pairs in a single SAPI driver run.
        """
        if not items:
            return []

        resolved_items: List[Tuple[str, Path]] = []
        for text, path in items:
            target_path = Path(path).resolve()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_items.append((text.strip(), target_path))

        try:
            # Initialize fresh driver for the batch
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            # Queue all synthesis targets into SAPI driver
            for text, target_path in resolved_items:
                if not text:
                    text = "..."
                engine.save_to_file(text, str(target_path))

            # Run message pump once to process the entire queue
            engine.runAndWait()
            del engine

            durations: List[float] = []
            for text, target_path in resolved_items:
                if not target_path.is_file() or target_path.stat().st_size == 0:
                    raise RuntimeError(f"Audio file was not generated: {target_path}")
                dur = get_audio_duration_seconds(target_path)
                durations.append(dur)
                logger.debug(f"Synthesized '{target_path.name}' ({dur}s) for text: '{text[:35]}...'")

            return durations

        except Exception as e:
            logger.error(f"Error during pyttsx3 batch synthesis: {e}")
            raise RuntimeError(f"pyttsx3 batch synthesis failed: {e}") from e
