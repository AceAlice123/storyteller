"""
Storyteller - Automated Text-to-Video Synthesis Platform
Module 3: CLIP Machine Learning Visual Aligner
Course: IGNOU MCA MCSP-232

Responsible for:
1. Scanning local image assets and managing SQLite embedding caching (512-d tensors).
2. Deep learning multimodal inference via OpenAI CLIP (ViT-B/32).
3. Mathematical cosine similarity matrix calculation: (text_vector . image_vectors^T).
4. Applying PyTorch softmax across tensor dimensions to compute selection probabilities.
5. Associating winning visual assets with sequence records in the image_assets table.
6. Triggering memory cleanup prior to video compositing.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from database.repository import StorytellerRepository
from modules.aligner.clip_model import CLIPModelWrapper, CLIPModelFactory
from utils.helpers import scan_image_directory
from utils.logger import setup_logger
from config import CLIP_BATCH_SIZE

logger = setup_logger("Storyteller.VisualAligner")


class VisualAligner:
    """
    Multimodal visual alignment engine using CLIP ViT-B/32.
    Integrates persistent SQLite embedding caching to ensure high scalability and fast
    re-runs on large image repositories.
    """

    def __init__(
        self,
        repository: StorytellerRepository,
        clip_wrapper: Optional[CLIPModelWrapper] = None,
        use_mock: bool = False
    ):
        self.repository = repository
        self.clip_wrapper = clip_wrapper or CLIPModelFactory.get_model(use_mock=use_mock)

    def compute_image_catalog_embeddings(
        self,
        image_paths: List[Path],
        force_recompute: bool = False
    ) -> Tuple[List[Path], np.ndarray]:
        """
        Computes 512-d CLIP embeddings for a list of image paths, utilizing the SQLite cache.

        Args:
            image_paths: List of filesystem paths to image files.
            force_recompute: If True, bypasses database cache and recalculates.

        Returns:
            Tuple[List[Path], np.ndarray]: (valid_image_paths, embeddings_matrix of shape [N, 512])
        """
        valid_paths: List[Path] = []
        embeddings_list: List[np.ndarray] = []
        missing_paths: List[Path] = []
        missing_indices: List[int] = []

        logger.info(f"Checking embedding cache for {len(image_paths)} visual assets...")

        for idx, img_path in enumerate(image_paths):
            stat = img_path.stat()
            mtime = stat.st_mtime
            size = stat.st_size

            cached_emb = None if force_recompute else self.repository.get_cached_embedding(
                str(img_path), mtime, size
            )

            if cached_emb is not None:
                valid_paths.append(img_path)
                embeddings_list.append(cached_emb)
            else:
                missing_paths.append(img_path)
                missing_indices.append(idx)

        logger.info(
            f"Embedding Cache Summary: {len(embeddings_list)} hits, {len(missing_paths)} misses."
        )

        # Compute embeddings for missing images in batches
        if missing_paths:
            logger.info(f"Computing CLIP embeddings for {len(missing_paths)} images...")
            self.clip_wrapper.load()

            for i in range(0, len(missing_paths), CLIP_BATCH_SIZE):
                batch_paths = missing_paths[i : i + CLIP_BATCH_SIZE]
                batch_pil_imgs: List[Image.Image] = []
                batch_valid_paths: List[Path] = []

                for p in batch_paths:
                    try:
                        with Image.open(p) as img:
                            # Convert to standard RGB format (handles CMYK, RGBA, Grayscale)
                            rgb_img = img.convert("RGB")
                            batch_pil_imgs.append(rgb_img)
                            batch_valid_paths.append(p)
                    except Exception as e:
                        logger.warning(f"Could not load image file {p}: {e}. Skipping.")

                if not batch_pil_imgs:
                    continue

                # Encode batch via CLIP
                batch_tensors = self.clip_wrapper.encode_images(batch_pil_imgs)
                batch_numpy = batch_tensors.numpy()

                for path_obj, emb_vec in zip(batch_valid_paths, batch_numpy):
                    stat = path_obj.stat()
                    # Persist to SQLite cache
                    self.repository.save_cached_embedding(
                        str(path_obj), stat.st_mtime, stat.st_size, emb_vec
                    )
                    valid_paths.append(path_obj)
                    embeddings_list.append(emb_vec)

        if not embeddings_list:
            raise ValueError("No valid image embeddings could be obtained.")

        # Stack into unified (N, 512) feature matrix
        embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
        logger.info(
            f"Image embedding catalog ready: {embeddings_matrix.shape[0]} assets, "
            f"dimension {embeddings_matrix.shape[1]}."
        )
        return valid_paths, embeddings_matrix

    def align_project_sequences(
        self,
        project_id: int,
        image_dir: Path | str,
        force_recompute: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic matching between project sentences and the image repository.

        Args:
            project_id: Target project primary key.
            image_dir: Directory containing candidate images.
            force_recompute: Whether to bypass database caching.

        Returns:
            List[Dict[str, Any]]: Alignment details including matched paths and scores.
        """
        image_paths = scan_image_directory(image_dir)
        valid_paths, image_embeddings_np = self.compute_image_catalog_embeddings(
            image_paths=image_paths,
            force_recompute=force_recompute
        )

        sequences = self.repository.get_sequences(project_id)
        if not sequences:
            raise ValueError(f"No sequence records found for Project ID: {project_id}")

        logger.info(
            f"Aligning {len(sequences)} text sequences against {len(valid_paths)} images..."
        )
        self.clip_wrapper.load()

        # Convert image embeddings to PyTorch tensor for matrix multiplication
        image_embeddings_t = torch.from_numpy(image_embeddings_np)  # Shape: (N, 512)

        script_texts = [s["script_text"] for s in sequences]
        text_embeddings_t = self.clip_wrapper.encode_text(script_texts)  # Shape: (M, 512)

        alignment_results: List[Dict[str, Any]] = []

        # Cosine similarity matrix: S = text_embeddings @ image_embeddings.T
        # Both embeddings are L2 normalized, so dot product is exact cosine similarity in [-1, 1]
        similarity_matrix = torch.matmul(text_embeddings_t, image_embeddings_t.T)  # Shape: (M, N)

        # Scale logits with CLIP standard temperature scaling (~100.0) before softmax
        temperature = 0.05
        softmax_probs = F.softmax(similarity_matrix / temperature, dim=-1)

        for idx, seq in enumerate(sequences):
            seq_id = seq["seq_id"]
            seq_order = seq["sequence_order"]
            text = seq["script_text"]

            similarities = similarity_matrix[idx]
            probabilities = softmax_probs[idx]

            # Highest similarity index
            best_idx = int(torch.argmax(similarities).item())
            best_image_path = valid_paths[best_idx]
            cosine_score = float(similarities[best_idx].item())
            softmax_confidence = float(probabilities[best_idx].item())

            # Log to SQLite image_assets table
            image_id = self.repository.insert_image_asset(
                seq_id=seq_id,
                file_path=str(best_image_path),
                clip_score=cosine_score
            )

            result = {
                "image_id": image_id,
                "seq_id": seq_id,
                "sequence_order": seq_order,
                "script_text": text,
                "image_file_path": str(best_image_path),
                "clip_score": round(cosine_score, 4),
                "confidence": round(softmax_confidence, 4)
            }
            alignment_results.append(result)
            logger.info(
                f"[Seq {seq_order:02d}] Matched -> '{best_image_path.name}' "
                f"(Cosine: {cosine_score:.3f}, Softmax Conf: {softmax_confidence*100:.1f}%)"
            )

        # Release ML memory prior to video compositing phase
        self.clip_wrapper.release_memory()

        return alignment_results
