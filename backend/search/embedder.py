"""
Text Embedding module using CLIP model.

Provides text-to-vector conversion using OpenAI's CLIP model
for semantic search capabilities.
"""

import logging
from typing import Union, List
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    CLIP-based text embedder for semantic search.
    
    Converts text queries into 512-dimensional embeddings using
    OpenAI's CLIP model (ViT-B/32 variant).
    
    Usage:
        embedder = TextEmbedder()
        vector = embedder.embed_single("woman on a train")
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the text embedder.
        
        Args:
            model_name: HuggingFace model identifier for CLIP
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        logger.info(f"TextEmbedder initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load the CLIP model on first use."""
        if self.model is None:
            logger.info(f"Loading CLIP model: {self.model_name}")
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self.model = CLIPTextModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text input(s).
        
        Args:
            text: Single text string or list of text strings
        
        Returns:
            numpy array of embeddings (512-d, L2-normalized)
            Shape: (512,) for single text, (N, 512) for batch
        """
        self._load_model()
        
        # Handle single string
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=77
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output  # [batch_size, 512]
        
        # L2 normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embeddings = embeddings.cpu().numpy()
        
        # Return single vector if single input
        if len(embeddings) == 1:
            return embeddings[0]
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Convenience method to embed a single text string.
        
        Args:
            text: Text string to embed
        
        Returns:
            512-dimensional embedding vector (L2-normalized)
        """
        return self.embed_text(text)
