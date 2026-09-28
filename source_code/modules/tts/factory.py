"""
Storyteller - Automated Text-to-Video Synthesis Platform
Factory Pattern: TTSEngineFactory

Responsible for instantiating the appropriate BaseTTSEngine strategy based on
configuration flags or runtime requirements.
"""

from typing import Dict, Type
from modules.tts.base import BaseTTSEngine
from modules.tts.pyttsx3_engine import Pyttsx3Engine
from modules.tts.dummy_engine import DummyTTSEngine
from utils.logger import setup_logger

logger = setup_logger("Storyteller.TTSFactory")


class TTSEngineFactory:
    """
    Factory for producing concrete BaseTTSEngine instances.
    Provides easy extensibility for future cloud or neural TTS engines (e.g. Piper, ElevenLabs).
    """

    _registry: Dict[str, Type[BaseTTSEngine]] = {
        "pyttsx3": Pyttsx3Engine,
        "dummy": DummyTTSEngine,
        "mock": DummyTTSEngine,
    }

    @classmethod
    def register_engine(cls, name: str, engine_class: Type[BaseTTSEngine]) -> None:
        """Dynamically registers a new TTS engine strategy."""
        cls._registry[name.lower()] = engine_class
        logger.info(f"Registered new TTS strategy: '{name}'")

    @classmethod
    def create_engine(cls, engine_name: str = "pyttsx3", **kwargs) -> BaseTTSEngine:
        """
        Creates and returns a concrete TTS engine instance.

        Args:
            engine_name: Identifier for the engine ('pyttsx3', 'dummy', etc.).
            **kwargs: Extra arguments passed to the engine constructor.

        Returns:
            BaseTTSEngine: Instantiated speech engine.

        Raises:
            ValueError: If the engine name is not registered.
        """
        engine_key = engine_name.lower().strip()
        if engine_key not in cls._registry:
            valid_engines = list(cls._registry.keys())
            raise ValueError(
                f"Unknown TTS engine '{engine_name}'. Supported engines: {valid_engines}"
            )

        engine_cls = cls._registry[engine_key]
        logger.debug(f"Instantiating TTS Engine: {engine_cls.__name__}")
        return engine_cls(**kwargs)
