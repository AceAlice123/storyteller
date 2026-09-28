"""
Storyteller - Integration Tests for StorytellerFacade (End-to-End Pipeline)
Tests full flow: Text Parser -> Audio Synthesizer -> CLIP ML Aligner -> Video Compositor.
"""

import tempfile
from pathlib import Path
from PIL import Image
import pytest

from config import PipelineConfig
from orchestrator.facade import StorytellerFacade
from database.repository import StorytellerRepository


def _create_sample_image(path: Path, color: tuple, size=(640, 480)) -> Path:
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


def test_facade_end_to_end_pipeline():
    """Verify complete execution of StorytellerFacade across all 4 modules."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        script_file = tmp_path / "script.txt"
        img_dir = tmp_path / "images"
        out_video = tmp_path / "output" / "final_story.mp4"
        db_path = tmp_path / "storyteller.db"
        temp_audio = tmp_path / "audio"

        img_dir.mkdir()

        # Write test script
        script_text = (
            "Artificial Intelligence represents a major breakthrough in computer science.\n\n"
            "Deep learning allows machines to comprehend the visual world around us."
        )
        script_file.write_text(script_text, encoding="utf-8")

        # Create sample images
        _create_sample_image(img_dir / "ai_robot.jpg", (40, 100, 200))
        _create_sample_image(img_dir / "code_matrix.png", (20, 180, 50))
        _create_sample_image(img_dir / "vintage_hardware.jpg", (180, 120, 40))

        config = PipelineConfig(
            script_path=script_file,
            image_dir=img_dir,
            output_path=out_video,
            project_name="Facade Integration Test",
            db_path=db_path,
            temp_audio_dir=temp_audio,
            tts_engine="dummy",  # Use fast dummy for automated testing
            dry_run=False
        )

        facade = StorytellerFacade()
        rendered_video = facade.run(config)

        # Assertions
        assert rendered_video.is_file()
        assert rendered_video.stat().st_size > 0

        # Check database records
        repo = StorytellerRepository(db_path=db_path)
        timeline = repo.get_project_timeline(project_id=1)
        assert len(timeline) == 2
        for item in timeline:
            assert Path(item["audio_file_path"]).exists()
            assert Path(item["image_file_path"]).exists()
            assert item["audio_duration"] > 0
            assert item["clip_score"] != 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
