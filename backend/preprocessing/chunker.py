import logging
from typing import List
import cv2

from models.metadata import VideoChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Chunker:
    """
    Simple fixed-interval chunking.
    """
    
    def __init__(self, chunk_duration: float = 10.0):
        """
        Args:
            chunk_duration: Target duration for each chunk in seconds
        """
        self.chunk_duration = chunk_duration
    
    def chunk_video(self, video_path: str, video_id: str) -> List[VideoChunk]:
        """
        Split video into fixed-duration chunks.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for this video
            
        Returns:
            List of VideoChunk objects
        """
        logger.info(f"Starting basic chunking for {video_path}")
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"Video duration: {duration:.1f}s, FPS: {fps}")
        
        # Create chunks
        chunks = []
        chunk_start = 0.0
        chunk_idx = 0
        
        while chunk_start < duration:
            chunk_end = min(chunk_start + self.chunk_duration, duration)
            
            chunk_id = f"{video_id}_chunk_{chunk_idx:04d}"
            chunk = VideoChunk(
                chunk_id=chunk_id,
                start_time=chunk_start,
                end_time=chunk_end
            )
            chunks.append(chunk)
            
            logger.info(f"Created {chunk}")
            
            chunk_start = chunk_end
            chunk_idx += 1
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

