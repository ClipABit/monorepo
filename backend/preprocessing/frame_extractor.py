import logging
from typing import Tuple, Optional
import cv2
import numpy as np

from models.metadata import VideoChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extract frames from video chunks at a fixed sampling rate.
    """
    
    def __init__(self, sampling_fps: float = 1.0):
        """
        Args:
            sampling_fps: How many frames per second to extract
        """
        self.sampling_fps = sampling_fps
    
    def extract_frames(self, video_path: str, chunk: VideoChunk) -> Tuple[np.ndarray, float]:
        """
        Extract frames from a specific chunk of video.
        
        Args:
            video_path: Path to video file
            chunk: The chunk to extract from
            
        Returns:
            (frames_array, actual_fps_used)
        """
        logger.info(f"Extracting frames from {chunk.chunk_id}")
        
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate which frames to extract
        start_frame = int(chunk.start_time * video_fps)
        end_frame = int(chunk.end_time * video_fps)
        
        # Sample every N frames based on sampling_fps
        frame_interval = max(1, int(video_fps / self.sampling_fps))
        
        frames = []
        frame_indices = []
        
        for frame_idx in range(start_frame, end_frame, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
                frame_indices.append(frame_idx)
            else:
                logger.warning(f"Failed to read frame {frame_idx}")
        
        cap.release()
        
        actual_fps = len(frames) / chunk.duration if chunk.duration > 0 else 0
        logger.info(f"Extracted {len(frames)} frames at {actual_fps:.2f} fps")
        
        return np.array(frames) if frames else np.array([]), actual_fps
    
    def extract_single_frame(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a single frame at a specific timestamp.
        Useful for generating thumbnails.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            
        Returns:
            Frame as numpy array, or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
