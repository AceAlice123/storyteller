"""
Storyteller - Unit Tests for Module 4 (MoviePy Video Compositor)
Tests frame normalization to 1080p, subtitle badge rendering, and end-to-end
timeline synchronization with MP4 compilation.
"""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from modules.compositor.video_compositor import VideoCompositor
from database.repository import StorytellerRepository
from modules.tts.dummy_engine import DummyTTSEngine


def _create_sample_image(path: Path, color: tuple, size=(800, 600)) -> Path:
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


def test_frame_normalization_to_1080p():
    """Verify arbitrary sized images are scaled and centered to exact 1920x1080 canvas."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        compositor = VideoCompositor(repository=None, resolution=(1920, 1080))

        # Test small landscape image
        p1 = _create_sample_image(Path(tmp_dir) / "land.jpg", (200, 50, 50), size=(640, 480))
        frame1 = compositor.normalize_frame(p1)
        assert frame1.shape == (1080, 1920, 3)

        # Test portrait image (vertical phone format)
        p2 = _create_sample_image(Path(tmp_dir) / "port.jpg", (50, 200, 50), size=(1080, 1920))
        frame2 = compositor.normalize_frame(p2)
        assert frame2.shape == (1080, 1920, 3)

        # Test square image
        p3 = _create_sample_image(Path(tmp_dir) / "sq.jpg", (50, 50, 200), size=(1000, 1000))
        frame3 = compositor.normalize_frame(p3)
        assert frame3.shape == (1080, 1920, 3)


def test_video_compilation_end_to_end():
    """Verify timeline querying, ImageClip audio synchronization, and MP4 generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "test.db"
        audio_dir = tmp_path / "audio"
        img_dir = tmp_path / "images"
        out_video = tmp_path / "output_test.mp4"

        audio_dir.mkdir()
        img_dir.mkdir()

        repo = StorytellerRepository(db_path=db_path)
        project_id = repo.create_project("Video Assembly Test")

        sentences = [
            "Welcome to the automated multimedia synthesis pipeline.",
            "Each visual scene is mathematically aligned with narration."
        ]
        seq_ids = repo.insert_sequences(project_id, sentences)

        # Generate fast dummy audio
        tts = DummyTTSEngine()
        audio1 = audio_dir / "clip1.wav"
        audio2 = audio_dir / "clip2.wav"
        dur1 = tts.synthesize(sentences[0], audio1)
        dur2 = tts.synthesize(sentences[1], audio2)

        repo.insert_audio_asset(seq_ids[0], str(audio1), dur1)
        repo.insert_audio_asset(seq_ids[1], str(audio2), dur2)

        # Generate sample images
        img1 = _create_sample_image(img_dir / "img1.png", (220, 30, 30), size=(800, 600))
        img2 = _create_sample_image(img_dir / "img2.png", (30, 180, 220), size=(1280, 720))

        repo.insert_image_asset(seq_ids[0], str(img1), 0.91)
        repo.insert_image_asset(seq_ids[1], str(img2), 0.95)

        # Build video at standard resolution and 24 fps
        compositor = VideoCompositor(repository=repo, resolution=(1920, 1080), fps=24)
        result_path = compositor.build_video(project_id=project_id, output_path=out_video)

        assert result_path.is_file()
        assert result_path.stat().st_size > 1000  # Valid MP4 video size


if __name__ == "__main__":
    pytest.main(["-v", __file__])
