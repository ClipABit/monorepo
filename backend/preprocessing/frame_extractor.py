import logging
from typing import Tuple, Optional
import cv2
import numpy as np

from models.metadata import VideoChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Adaptive frame extraction that adjusts sampling rate based on motion/complexity.
    High-motion scenes get more frames, static scenes get fewer frames.
    """

    def __init__(
        self,
        min_fps: float = 0.5,
        max_fps: float = 2.0,
        motion_threshold: float = 25.0
    ):
        """
        Args:
            min_fps: Minimum frames per second for static scenes
            max_fps: Maximum frames per second for high-motion scenes
            motion_threshold: Threshold for detecting significant motion (higher = less sensitive)
        """
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.motion_threshold = motion_threshold
        logger.debug("Initialized adaptive extractor: min_fps=%.1f, max_fps=%.1f, motion_threshold=%.1f",
                     min_fps, max_fps, motion_threshold)

    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate the difference between two frames using mean absolute difference.

        Args:
            frame1, frame2: Frames to compare

        Returns:
            Difference score (higher = more motion/change)
        """
        # Convert to grayscale for faster comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate mean absolute difference
        diff = cv2.absdiff(gray1, gray2)
        return float(np.mean(diff))

    def extract_frames(self, video_path: str, chunk: VideoChunk) -> Tuple[np.ndarray, float]:
        """
        Adaptively extract frames based on motion/complexity using streaming.

        Args:
            video_path: Path to video file
            chunk: The chunk to extract from

        Returns:
            (frames_array, actual_fps_used)
        """
        logger.debug("Adaptive extraction: chunk_id=%s, time_range=%.1f-%.1fs",
                     chunk.chunk_id, chunk.start_time, chunk.end_time)

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate frame range
        start_frame = int(chunk.start_time * video_fps)
        end_frame = int(chunk.end_time * video_fps)

        # Sampling intervals
        min_interval = max(1, int(video_fps / self.max_fps))
        max_interval = max(1, int(video_fps / self.min_fps))

        # Seek to start position once
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        prev_frame = None
        prev_gray = None
        current_idx = start_frame
        next_capture_idx = start_frame  # Track when to capture next frame

        # Stream through the chunk once
        while current_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame %d", current_idx)
                break

            # Check if we should capture this frame
            if current_idx >= next_capture_idx:
                if prev_frame is None:
                    # Always keep the first frame
                    frames.append(frame)
                    prev_frame = frame
                    # Convert to grayscale for next comparison
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    next_capture_idx += min_interval
                else:
                    # Calculate motion using cached grayscale
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(prev_gray, current_gray)
                    motion_score = float(np.mean(diff))

                    # Always keep the frame (we've decided to sample here)
                    frames.append(frame)

                    # Decide next interval based on motion
                    if motion_score > self.motion_threshold:
                        next_capture_idx += min_interval
                        logger.debug("High motion (score=%.1f) at frame %d, next in %d frames",
                                   motion_score, current_idx, min_interval)
                    else:
                        next_capture_idx += max_interval
                        logger.debug("Low motion (score=%.1f) at frame %d, next in %d frames",
                                   motion_score, current_idx, max_interval)

                    # Update previous frame for next comparison
                    prev_frame = frame
                    prev_gray = current_gray

            current_idx += 1

        cap.release()

        actual_fps = len(frames) / chunk.duration if chunk.duration > 0 else 0
        logger.info("Adaptive extraction complete: %d frames at avg %.2f fps (range: %.1f-%.1f fps)",
                    len(frames), actual_fps, self.min_fps, self.max_fps)

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
