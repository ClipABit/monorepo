"""
Text Embedder using CLIP for semantic search.

Handles loading CLIP model and generating normalized 512-d embeddings
for text queries and content.
"""

import logging
import torch
from functools import cache
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    CLIP-based text embedder for semantic search.
    
    Generates 512-dimensional L2-normalized embeddings using
    openai/clip-vit-base-patch32 model.
    
    Features:
    - Lazy model loading (on first use)
    - Batch processing support
    - GPU acceleration when available
    - L2 normalization for cosine similarity
    """
    
    MODEL_NAME = "openai/clip-vit-base-patch32"
    EMBEDDING_DIM = 512
    
    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"TextEmbedder initialized (device: {self._device})")
    
    def _load_model(self):
        """Lazy load CLIP model on first use."""
        if self._model is None:
            logger.info(f"Loading CLIP model: {self.MODEL_NAME}")
            self._model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
            self._model.eval()  # Set to evaluation mode
            logger.info("CLIP model loaded successfully")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate CLIP embeddings for text input.
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            numpy array of shape (n_texts, 512) with L2-normalized embeddings
        """
        self._load_model()
        
        # Ensure input is a list
        if isinstance(text, str):
            text = [text]
        
        logger.debug(f"Embedding {len(text)} text input(s)")
        
        # Process text through CLIP
        inputs = self._processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP's max token length
        ).to(self._device)
        
        # Generate embeddings
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            # L2 normalize for cosine similarity
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        embeddings = text_features.cpu().numpy()
        logger.debug(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Convenience method for embedding a single text.
        
        Args:
            text: Single string to embed
            
        Returns:
            numpy array of shape (512,) - single embedding vector
        """
        return self.embed_text(text)[0]
    
    @property
    def device(self) -> str:
        """Return the device being used (cpu/cuda)."""
        return self._device
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded in memory."""
        return self._model is not None
