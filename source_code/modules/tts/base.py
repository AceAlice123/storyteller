"""
Storyteller - Automated Text-to-Video Synthesis Platform
Strategy Pattern Interface for Text-to-Speech (TTS) Engines

Enables modular, pluggable speech synthesis without coupling the orchestrator
to any single vendor or library.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple


class BaseTTSEngine(ABC):
    """
    Abstract Strategy class defining the contract for all TTS synthesis engines.
    Any new speech engine (e.g., pyttsx3, Piper, Coqui, ElevenLabs) can be plugged in
    simply by implementing this interface.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Returns the unique human-readable name of the engine."""
        pass

    @abstractmethod
    def synthesize_batch(self, items: List[Tuple[str, Path | str]]) -> List[float]:
        """
        Synthesizes a batch of text items into corresponding audio files.
        Enables efficient queueing and avoids OS event loop lockups.

        Args:
            items: List of (text_string, output_wav_path) tuples.

        Returns:
            List[float]: Exact duration of each generated audio file in seconds.
        """
        pass

    def synthesize(self, text: str, output_path: Path | str) -> float:
        """
        Convenience method to synthesize a single audio segment.
        Delegates directly to synthesize_batch.
        """
        durations = self.synthesize_batch([(text, output_path)])
        return durations[0]
