import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor
)


class VideoEmbedder:
    """
    A class to handle video embedding generation using various models.
    """
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_model = None
        self._clip_processor = None  
        self._get_clip_model()
        
    def _get_clip_model(self):
        """Lazily load and return CLIP model + processor."""
        if self._clip_model is None or self._clip_processor is None:
            print("Loading CLIP model into memory...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
                self._device
            )
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_fast=True  # uses fast tokenizer implemented in Rust 
            )
        return self._clip_model, self._clip_processor
   
    def _generate_clip_embedding(self, frames, num_frames: int = 8) -> torch.Tensor:
        """
        Generate a single embedding for a video chunk by averaging the normalized
        embeddings of sampled frames using the Open AI CLIP Model.
        Args:
            processed_chunk (Dict[str, Any]): The processed video chunk object.
            num_frames (int): Number of frames to sample evenly across the video.
        
        Returns:
            torch.Tensor: A single, normalized embedding tensor for the video chunk.
        """
        
        # Fetch the preloaded model and processor
        model, processor = self._get_clip_model()
        
        # Sample frames evenly across the video if the num frames is greater than available frames 
        num_frames = min(num_frames, frames.shape[0])
        frame_indices = np.linspace(0, frames.shape[0] - 1, num_frames).astype(int)
        sampled_frames = [Image.fromarray(frames[idx]) for idx in frame_indices]
        
        # Transform the frame data to match the standard dimensions and normalization of the pixel values to the ranges 
        # of the data the model was trained on.
        inputs = processor(images=sampled_frames, return_tensors="pt", size=224).to(self._device)    
        
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            frame_features = output.pooler_output if hasattr(output, 'pooler_output') else output
            frame_features = frame_features / frame_features.norm(p=2, dim=-1, keepdim=True)
            
            video_embedding = frame_features.mean(dim=0)
            video_embedding = video_embedding / video_embedding.norm(p=2, dim=-1, keepdim=True)
            
        
        return video_embedding.cpu()    
    