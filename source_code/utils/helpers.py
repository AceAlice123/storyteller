"""
Storyteller - Automated Text-to-Video Synthesis Platform
Helper Functions Module

Contains utility functions for audio metadata extraction, filesystem security,
path sanitization, and asset directory validation.
"""

from pathlib import Path
from typing import List
import os
import wave
import contextlib
from config import SUPPORTED_IMAGE_EXTENSIONS
from utils.logger import setup_logger

logger = setup_logger("Storyteller.Helpers")


def get_audio_duration_seconds(audio_path: Path | str) -> float:
    """
    Calculates the exact duration in seconds of a standard WAV audio file
    using Python's built-in wave module without external process overhead.

    Args:
        audio_path: Absolute or relative Path to the .wav file.

    Returns:
        float: Duration of audio in seconds.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid readable WAV file.
    """
    path = Path(audio_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wave_file:
            frames = wave_file.getnframes()
            rate = wave_file.getframerate()
            duration = frames / float(rate)
            return round(duration, 4)
    except Exception as e:
        logger.error(f"Failed to inspect WAV duration for {path}: {e}")
        raise ValueError(f"Unable to read audio metadata from {path}: {e}") from e


def scan_image_directory(directory: Path | str) -> List[Path]:
    """
    Recursively scans a directory for supported image assets (.jpg, .jpeg, .png, .webp).
    Implements Directory Traversal Mitigation (Section 14.2 of project synopsis)
    by ensuring resolved paths remain within or relative to the validated root.

    Args:
        directory: Directory to search for visual assets.

    Returns:
        List[Path]: Sorted list of resolved Path objects for all valid images.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
        ValueError: If no valid image files are discovered.
    """
    root_dir = Path(directory).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {root_dir}")

    valid_images: List[Path] = []
    for item in root_dir.rglob("*"):
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            # Prevent directory traversal attacks
            try:
                # Ensure the resolved file path starts with the resolved directory root
                item.resolve().relative_to(root_dir)
                valid_images.append(item.resolve())
            except ValueError:
                logger.warning(f"Skipping file outside allowed directory boundaries: {item}")

    # Deterministic sorting by filename
    valid_images.sort()

    if not valid_images:
        raise ValueError(
            f"No valid image files found in '{root_dir}' with extensions {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    logger.info(f"Discovered {len(valid_images)} valid image assets in '{root_dir}'.")
    return valid_images


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    Converts arbitrary strings into safe filenames by replacing non-alphanumeric chars.

    Args:
        name: Raw input string.
        max_length: Maximum allowed filename length.

    Returns:
        str: Sanitized filesystem-safe name.
    """
    clean_chars = [c if c.isalnum() or c in ("-", "_") else "_" for c in name]
    sanitized = "".join(clean_chars).strip("_")
    return sanitized[:max_length] if sanitized else "asset"
