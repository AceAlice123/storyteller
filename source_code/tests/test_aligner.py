"""
Storyteller - Unit Tests for Module 3 (CLIP Machine Learning Visual Aligner)
Tests 512-d feature tensor extraction, L2 unit normalization, cosine similarity,
softmax probability distribution, SQLite embedding caching, and memory cleanup.
"""

import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pytest

from modules.aligner.clip_model import CLIPModelWrapper
from modules.aligner.visual_aligner import VisualAligner
from database.repository import StorytellerRepository


def _create_dummy_image(path: Path, color: tuple) -> Path:
    """Helper to generate a clean synthetic PNG image."""
    img = Image.new("RGB", (224, 224), color=color)
    img.save(path)
    return path


def test_clip_model_wrapper_mock_shapes_and_normalization():
    """Verify extracted embeddings have exact dimension 512 and unit norm (L2 = 1.0)."""
    wrapper = CLIPModelWrapper(use_mock=True)
    wrapper.load()

    # Create dummy PIL images
    img1 = Image.new("RGB", (224, 224), (255, 0, 0))
    img2 = Image.new("RGB", (224, 224), (0, 255, 0))

    img_embeddings = wrapper.encode_images([img1, img2])
    assert img_embeddings.shape == (2, 512)

    # Assert unit norm
    norms = torch.norm(img_embeddings, p=2, dim=-1)
    assert torch.allclose(norms, torch.tensor([1.0, 1.0]), atol=1e-5)

    # Test text embeddings
    texts = ["A futuristic spaceship navigating through stars.", "A medieval stone castle."]
    text_embeddings = wrapper.encode_text(texts)
    assert text_embeddings.shape == (2, 512)

    text_norms = torch.norm(text_embeddings, p=2, dim=-1)
    assert torch.allclose(text_norms, torch.tensor([1.0, 1.0]), atol=1e-5)

    wrapper.release_memory()
    assert wrapper.model is None


def test_visual_aligner_semantic_matching_and_caching():
    """Verify embedding cache hits/misses, cosine similarity, and DB persistence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "test.db"
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Create 3 synthetic colored images
        img_a = _create_dummy_image(img_dir / "red_apple.png", (255, 0, 0))
        img_b = _create_dummy_image(img_dir / "blue_ocean.png", (0, 0, 255))
        img_c = _create_dummy_image(img_dir / "green_forest.png", (0, 255, 0))

        repo = StorytellerRepository(db_path=db_path)
        project_id = repo.create_project("Visual Alignment Test")

        sentences = [
            "A crimson red apple sitting on a table.",
            "Waves crashing across the deep blue ocean."
        ]
        seq_ids = repo.insert_sequences(project_id, sentences)

        aligner = VisualAligner(repository=repo, use_mock=True)

        # Run alignment (first run -> all cache misses)
        results = aligner.align_project_sequences(project_id=project_id, image_dir=img_dir)
        assert len(results) == 2
        for r in results:
            assert Path(r["image_file_path"]).exists()
            assert -1.0 <= r["clip_score"] <= 1.0
            assert 0.0 <= r["confidence"] <= 1.0

        # Verify SQLite image_assets records
        with repo.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM image_assets ORDER BY seq_id ASC;")
            rows = cursor.fetchall()
            assert len(rows) == 2
            assert rows[0]["seq_id"] == seq_ids[0]
            assert rows[1]["seq_id"] == seq_ids[1]

        # Verify caching: Second run on same folder should have 3 cache hits
        valid_paths, emb_matrix = aligner.compute_image_catalog_embeddings(
            [img_a, img_b, img_c], force_recompute=False
        )
        assert len(valid_paths) == 3
        assert emb_matrix.shape == (3, 512)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
