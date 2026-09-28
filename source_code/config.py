"""
Storyteller - Automated Text-to-Video Synthesis Platform
Configuration Module

This module defines centralized configuration parameters for the Storyteller pipeline,
including directory paths, SQLite database settings, Hugging Face model identifiers,
multimedia rendering constraints, and logging configurations.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import torch


# Base Paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATABASE_PATH: Path = BASE_DIR / "storyteller.db"
SCHEMA_PATH: Path = BASE_DIR / "database" / "schema.sql"
DEFAULT_TEMP_AUDIO_DIR: Path = BASE_DIR / "temp_audio"
DEFAULT_OUTPUT_DIR: Path = BASE_DIR / "output"

# Machine Learning / CLIP Configuration
CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
EMBEDDING_DIMENSION: int = 512
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_BATCH_SIZE: int = 16

# Supported Image Formats
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# Text-to-Speech (TTS) Configuration
DEFAULT_TTS_ENGINE: str = "pyttsx3"
TTS_SPEECH_RATE: int = 165  # Words per minute
TTS_VOLUME: float = 1.0     # 0.0 to 1.0

# Video Compositor Configuration
VIDEO_RESOLUTION: tuple[int, int] = (1920, 1080)  # Standard Full HD 1080p (width, height)
VIDEO_FPS: int = 24
VIDEO_CODEC: str = "libx264"
AUDIO_CODEC: str = "aac"

# Subtitle Configuration
SUBTITLE_FONT_SIZE: int = 38
SUBTITLE_FONT_COLOR: str = "white"
SUBTITLE_BG_COLOR: tuple[int, int, int, int] = (0, 0, 0, 180)  # RGBA translucent black
SUBTITLE_BOTTOM_MARGIN: int = 60


@dataclass
class PipelineConfig:
    """
    Data class encapsulating runtime configuration for a specific video synthesis job.
    Enables dependency injection and flexible overrides during CLI execution or testing.
    """
    script_path: Path
    image_dir: Path
    output_path: Path = DEFAULT_OUTPUT_DIR / "final_story.mp4"
    project_name: str = "Default Storyteller Run"
    db_path: Path = DATABASE_PATH
    temp_audio_dir: Path = DEFAULT_TEMP_AUDIO_DIR
    tts_engine: str = DEFAULT_TTS_ENGINE
    device: str = DEVICE
    force_cache_recompute: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        """Validate input paths and parameters to prevent runtime failures."""
        if not self.script_path.exists():
            raise FileNotFoundError(f"Input script file does not exist: {self.script_path}")
        if not self.script_path.is_file():
            raise ValueError(f"Script path is not a file: {self.script_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Image path is not a directory: {self.image_dir}")

        # Ensure output directories exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)
