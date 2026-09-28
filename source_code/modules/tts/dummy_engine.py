"""
Storyteller - Automated Text-to-Video Synthesis Platform
Mock / Dummy TTS Strategy

Generates synthetic audio PCM WAV files for high-speed testing, automated CI/CD pipelines,
and environments without sound hardware drivers.
"""

from pathlib import Path
from typing import List, Tuple
import math
import struct
import wave

from modules.tts.base import BaseTTSEngine
from utils.helpers import get_audio_duration_seconds
from utils.logger import setup_logger

logger = setup_logger("Storyteller.DummyEngine")


class DummyTTSEngine(BaseTTSEngine):
    """
    Mock TTS Strategy generating real playable WAV files without relying on external drivers.
    Calculates duration proportional to sentence length (~2.5 words per second).
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    @property
    def engine_name(self) -> str:
        return "Dummy/Mock Synthesizer"

    def synthesize_batch(self, items: List[Tuple[str, Path | str]]) -> List[float]:
        """
        Synthesizes multiple mock audio files in sequence.
        """
        durations: List[float] = []
        for text, path in items:
            dur = self._synthesize_single(text, path)
            durations.append(dur)
        return durations

    def _synthesize_single(self, text: str, output_path: Path | str) -> float:
        """Generates a single PCM mono WAV file."""
        target_path = Path(output_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        words = text.strip().split()
        estimated_duration = max(1.5, round(len(words) / 2.5, 2))
        total_samples = int(self.sample_rate * estimated_duration)

        with wave.open(str(target_path), "wb") as wav_out:
            wav_out.setnchannels(1)       # Mono
            wav_out.setsampwidth(2)       # 16-bit PCM
            wav_out.setframerate(self.sample_rate)

            samples = bytearray()
            for i in range(total_samples):
                t = float(i) / self.sample_rate
                amplitude = 4000.0 * min(1.0, t * 10, (estimated_duration - t) * 10)
                val = int(amplitude * math.sin(2.0 * math.pi * 440.0 * t))
                samples.extend(struct.pack("<h", max(-32768, min(32767, val))))

            wav_out.writeframes(samples)

        duration = get_audio_duration_seconds(target_path)
        logger.debug(f"Generated dummy audio '{target_path.name}' ({duration}s)")
        return duration
