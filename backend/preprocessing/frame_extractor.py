import logging
from typing import Tuple, Optional, Dict
import cv2
import numpy as np

from models.metadata import VideoChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Adaptive frame extraction that adjusts sampling rate based on motion/complexity.
    High-motion scenes get more frames, static scenes get fewer frames.
    Also calculates complexity scores for each chunk.
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
        self.motion_scores = []  # Track motion scores for complexity calculation
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

    def extract_frames(self, video_path: str, chunk: VideoChunk) -> Tuple[np.ndarray, float, float]:
        """
        Adaptively extract frames based on motion/complexity using streaming.

        Args:
            video_path: Path to video file
            chunk: The chunk to extract from

        Returns:
            (frames_array, actual_fps_used, complexity_score)
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
        motion_scores = []
        edge_densities = []
        color_variances = []

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

                    # Calculate complexity metrics for first frame
                    edge_densities.append(self._calculate_edge_density(prev_gray))
                    color_variances.append(self._calculate_color_variance(frame))

                    next_capture_idx += min_interval
                else:
                    # Calculate motion using cached grayscale
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(prev_gray, current_gray)
                    motion_score = float(np.mean(diff))
                    motion_scores.append(motion_score)

                    # Always keep the frame (we've decided to sample here)
                    frames.append(frame)

                    # Calculate complexity metrics
                    edge_densities.append(self._calculate_edge_density(current_gray))
                    color_variances.append(self._calculate_color_variance(frame))

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

        # Calculate overall complexity score
        complexity_score = self._calculate_complexity_score(
            motion_scores, edge_densities, color_variances
        )

        actual_fps = len(frames) / chunk.duration if chunk.duration > 0 else 0
        logger.info("Adaptive extraction complete: %d frames at avg %.2f fps (range: %.1f-%.1f fps), complexity=%.3f",
                    len(frames), actual_fps, self.min_fps, self.max_fps, complexity_score)

        return np.array(frames) if frames else np.array([]), actual_fps, complexity_score

    def _calculate_edge_density(self, gray_frame: np.ndarray) -> float:
        """
        Calculate edge density using Canny edge detection.
        Higher values indicate more detail/complexity.
        """
        edges = cv2.Canny(gray_frame, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return float(edge_density)

    def _calculate_color_variance(self, frame: np.ndarray) -> float:
        """
        Calculate color variance across the frame.
        Higher values indicate more colorful/varied scenes.
        """
        # Calculate standard deviation across all channels
        std_dev = np.std(frame)
        return float(std_dev)

    def _calculate_complexity_score(
        self,
        motion_scores: list,
        edge_densities: list,
        color_variances: list
    ) -> float:
        """
        Calculate overall complexity score from multiple metrics.

        Combines:
        - Motion (30%): How much the scene moves
        - Edge density (40%): How much detail/texture
        - Color variance (30%): How varied the colors are

        Returns score normalized to 0-1 range.
        """
        if not edge_densities:
            return 0.0

        # Calculate average metrics
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        avg_edge_density = np.mean(edge_densities)
        avg_color_variance = np.mean(color_variances)

        # Normalize each metric (approximate ranges)
        # Motion: 0-50 typical range
        # Edge density: 0-0.3 typical range
        # Color variance: 0-80 typical range
        motion_normalized = min(avg_motion / 50.0, 1.0)
        edge_normalized = min(avg_edge_density / 0.3, 1.0)
        color_normalized = min(avg_color_variance / 80.0, 1.0)

        # Weighted combination
        complexity = (
            0.30 * motion_normalized +
            0.40 * edge_normalized +
            0.30 * color_normalized
        )

        return float(complexity)
    
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
