"""
Storyteller - Automated Text-to-Video Synthesis Platform
Factory & Wrapper: CLIP Model Wrapper (OpenAI ViT-B/32)
Course: IGNOU MCA MCSP-232

Responsible for:
1. Hardware acceleration detection (CUDA vs CPU).
2. Loading pretrained CLIP model and processor from Hugging Face.
3. Extracting 512-dimensional normalized image and text embeddings.
4. Rigorous GPU/RAM memory management (deallocating tensors, empty_cache, gc.collect).
"""

from typing import List, Optional
from pathlib import Path
import gc
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CLIP_MODEL_NAME, EMBEDDING_DIMENSION, DEVICE
from utils.logger import setup_logger

logger = setup_logger("Storyteller.CLIPModel")


class CLIPModelWrapper:
    """
    Encapsulates OpenAI CLIP ViT-B/32 architecture for multimodal feature extraction.
    Provides batch encoding for images and text with automatic L2-normalization.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        device: Optional[str] = None,
        use_mock: bool = False
    ):
        self.model_name = model_name
        self.device = torch.device(device or DEVICE)
        self.use_mock = use_mock
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Loads the CLIP model and processor into memory and transfers to target device.
        """
        if self._is_loaded:
            return

        if self.use_mock:
            logger.info("Initializing CLIPModelWrapper in deterministic Mock Mode.")
            self._is_loaded = True
            return

        logger.info(f"Loading CLIP model '{self.model_name}' on target device: {self.device}...")
        try:
            from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Freeze weights for inference mode
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self._is_loaded = True
            logger.info(f"Successfully loaded CLIP model on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face CLIP model: {e}")
            raise RuntimeError(f"Could not load CLIP model '{self.model_name}': {e}") from e

    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extracts 512-dimensional normalized image embeddings for a batch of PIL images.

        Args:
            images: List of PIL Image objects in RGB format.

        Returns:
            torch.Tensor: Shape (N, 512) tensor of normalized image feature vectors.
        """
        if not self._is_loaded:
            self.load()

        if self.use_mock:
            return self._mock_encode_images(images)

        # Preprocess visual inputs using CLIP processor
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Ensure float tensor and normalize to unit hypersphere (L2-norm = 1.0)
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            image_features = F.normalize(image_features, p=2, dim=-1)

        return image_features.cpu()

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Extracts 512-dimensional normalized text embeddings for a list of strings.

        Args:
            texts: List of sentence strings.

        Returns:
            torch.Tensor: Shape (N, 512) tensor of normalized text feature vectors.
        """
        if not self._is_loaded:
            self.load()

        if self.use_mock:
            return self._mock_encode_text(texts)

        # Tokenize and pad text inputs
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu()

    def _mock_encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Deterministic mock image encoder producing unit 512-d vectors."""
        tensors = []
        for img in images:
            # Hash thumbnail bytes for deterministic embedding
            thumb = img.resize((32, 32)).convert("L")
            h = hashlib.sha256(thumb.tobytes()).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(EMBEDDING_DIMENSION).astype(np.float32)
            norm = np.linalg.norm(vec)
            tensors.append(torch.from_numpy(vec / (norm if norm > 0 else 1.0)))
        return torch.stack(tensors, dim=0)

    def _mock_encode_text(self, texts: List[str]) -> torch.Tensor:
        """Deterministic mock text encoder producing unit 512-d vectors."""
        tensors = []
        for txt in texts:
            h = hashlib.sha256(txt.strip().lower().encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(EMBEDDING_DIMENSION).astype(np.float32)
            norm = np.linalg.norm(vec)
            tensors.append(torch.from_numpy(vec / (norm if norm > 0 else 1.0)))
        return torch.stack(tensors, dim=0)

    def release_memory(self) -> None:
        """
        Explicit memory deallocation adhering to Section 12 of the synopsis.
        Frees PyTorch model weights and clears GPU VRAM before video rendering.
        """
        logger.info("Executing explicit memory cleanup: releasing CLIP model and cache...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._is_loaded = False
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared PyTorch CUDA cache.")


class CLIPModelFactory:
    """
    Factory Pattern for obtaining CLIPModelWrapper instances.
    """

    _instance: Optional[CLIPModelWrapper] = None

    @classmethod
    def get_model(
        cls,
        model_name: str = CLIP_MODEL_NAME,
        device: Optional[str] = None,
        use_mock: bool = False
    ) -> CLIPModelWrapper:
        """
        Returns a configured CLIPModelWrapper instance.
        """
        if cls._instance is None or cls._instance.use_mock != use_mock:
            cls._instance = CLIPModelWrapper(
                model_name=model_name,
                device=device,
                use_mock=use_mock
            )
        return cls._instance
