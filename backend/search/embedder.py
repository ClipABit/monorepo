import logging
from typing import Union, List
import numpy as np
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Text Embedding module using CLIP model.

Provides text-to-vector conversion using OpenAI's CLIP model
for semantic search capabilities.
"""


class TextEmbedder:
    """
    CLIP-based text embedder for semantic search.
    
    Converts text queries into 512-dimensional embeddings using
    OpenAI's CLIP text model (ViT-B/32 variant).
    
    Uses CLIPTextModelWithProjection for efficiency (loads only text encoder,
    not the full CLIP model with vision encoder).
    
    Usage:
        embedder = TextEmbedder()
        vector = embedder.embed_text("woman on a train")
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the text embedder.

        Args:
            model_name: HuggingFace model identifier for CLIP. 
                       Defaults to "openai/clip-vit-base-patch32".
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        logger.info(f"TextEmbedder initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load the CLIP text model on first use."""
        if self.model is None:
            logger.info(f"Loading CLIP text model: {self.model_name}")
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self.model = CLIPTextModelWithProjection.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("CLIP text model loaded successfully")
    
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
        
        # Tokenize inputs
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP's max sequence length
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            # CLIPTextModelWithProjection outputs already-projected features
            text_features = self.model(**inputs).text_embeds
            # L2 normalize (essential for cosine similarity search)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Convert to numpy
        embeddings = text_features.cpu().numpy()
        
        # Return single vector if single input
        if len(embeddings) == 1:
            return embeddings[0]
        
        return embeddings