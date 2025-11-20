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
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
        model, processor = self._get_clip_model()
        
        num_frames = min(num_frames, frames.shape[0])
        frame_indices = np.linspace(0, frames.shape[0] - 1, num_frames).astype(int)
        sampled_frames = [Image.fromarray(frames[idx]) for idx in frame_indices]
        
        inputs = processor(images=sampled_frames, return_tensors="pt", size=224).to(self._device)    
        
        with torch.no_grad():
            frame_features = model.get_image_features(**inputs)
            frame_features = frame_features / frame_features.norm(p=2, dim=-1, keepdim=True)
            
            video_embedding = frame_features.mean(dim=0)
            video_embedding = video_embedding / video_embedding.norm(p=2, dim=-1, keepdim=True)
            
        
        return video_embedding.cpu()    
    
    
    def generate_clip_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate a normalized text embedding using the Open AI CLIP Model.
        
        Args:
            text (str): The input text string.
        
        Returns:
            torch.Tensor: A normalized embedding tensor for the input text.
        """
        model, processor = self._get_clip_model()
        
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(self._device)
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        print(f"Generated CLIP embedding for text: {text[:30]}...")
        
        return text_features.cpu().numpy()
    


    