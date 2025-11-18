"""
Frame Embedding module using CLIP model.

Provides image-to-vector conversion using OpenAI's CLIP model
for semantic video search capabilities.

Mirrors the architecture of search.embedder.py but for images instead of text.
"""

import logging
from typing import Union, List
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameEmbedder:
    """
    CLIP-based frame embedder for semantic video search.
    
    Converts video frames into 512-dimensional embeddings using
    OpenAI's CLIP model (ViT-B/32 variant).
    
    Uses the same CLIP model as TextEmbedder to ensure text queries
    can match video frames in the same embedding space.
    
    Usage:
        embedder = FrameEmbedder()
        vector = embedder.embed_frame(pil_image)
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the frame embedder.
        
        Args:
            model_name: HuggingFace model identifier for CLIP
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        logger.info(f"FrameEmbedder initialized (device: {self.device})")
    
    def _load_model(self):
        """Lazy load the CLIP model on first use."""
        if self.model is None:
            logger.info(f"Loading CLIP model: {self.model_name}")
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully")
    
    def embed_frames(self, frames: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Generate embeddings for frame image(s).
        
        Args:
            frames: Single PIL Image or list of PIL Images
        
        Returns:
            numpy array of embeddings (512-d, L2-normalized)
            Shape: (512,) for single frame, (N, 512) for batch
        """
        self._load_model()
        
        # Handle single image
        if isinstance(frames, Image.Image):
            frames = [frames]
        
        # Process inputs using CLIPProcessor
        inputs = self.processor(
            images=frames,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate embeddings using get_image_features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # L2 normalize (same as text embeddings)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Convert to numpy
        embeddings = image_features.cpu().numpy()
        
        # Return single vector if single input
        if len(embeddings) == 1:
            return embeddings[0]
        
        return embeddings
    
    def embed_single(self, frame: Image.Image) -> np.ndarray:
        """
        Convenience method to embed a single frame.
        
        Args:
            frame: PIL Image to embed
        
        Returns:
            512-dimensional embedding vector (L2-normalized)
        """
        return self.embed_frames(frame)
    
    def embed_frame_array(self, frame_array: np.ndarray) -> np.ndarray:
        """
        Embed a numpy array frame (common format from OpenCV).
        
        Args:
            frame_array: numpy array in RGB format (H, W, 3)
        
        Returns:
            512-dimensional embedding vector (L2-normalized)
        """
        # Convert numpy array to PIL Image
        if frame_array.dtype != np.uint8:
            frame_array = (frame_array * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(frame_array)
        return self.embed_single(pil_image)
