import logging
from typing import Optional
import cv2
import numpy as np

from models.metadata import VideoChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Compressor:
    """
    Compress video frames for efficient storage and processing.
    """
    
    def __init__(
        self,
        target_height: int = 480,
        target_width: int = 640,
        quality: int = 85
    ):
        """
        Args:
            target_height: Target height for resized frames
            target_width: Target width for resized frames
            quality: JPEG compression quality (0-100)
        """
        self.target_height = target_height
        self.target_width = target_width
        self.quality = quality
    
    def compress_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Compress a single frame by resizing and quality reduction.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Compressed frame
        """
        # Resize frame
        resized = cv2.resize(
            frame,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )
        
        return resized
    
    def compress_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Compress multiple frames.
        
        Args:
            frames: Array of frames (n_frames, height, width, channels)
            
        Returns:
            Array of compressed frames
        """
        logger.info(f"Compressing {len(frames)} frames to {self.target_width}x{self.target_height}")
        
        compressed = []
        for frame in frames:
            compressed.append(self.compress_frame(frame))
        
        result = np.array(compressed)
        logger.info(f"Compressed frames shape: {result.shape}")
        
        return result
    
    def get_compression_ratio(self, original_shape: tuple, compressed_shape: tuple) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_shape: Shape of original frames
            compressed_shape: Shape of compressed frames
            
        Returns:
            Compression ratio
        """
        original_size = np.prod(original_shape)
        compressed_size = np.prod(compressed_shape)
        
        return original_size / compressed_size if compressed_size > 0 else 1.0
