import logging
from typing import Tuple, Optional
import cv2
import numpy as np

from models.metadata import VideoChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Adaptive frame extractor with motion detection and complexity scoring.

    Streams through video chunks once, sampling frames at variable rates based
    on inter-frame motion. Simultaneously calculates complexity metrics.
    """

    # Complexity metric normalization ranges (empirically determined)
    MOTION_RANGE = 50.0
    EDGE_DENSITY_RANGE = 0.3
    COLOR_VARIANCE_RANGE = 80.0

    # Complexity weights
    MOTION_WEIGHT = 0.30
    EDGE_WEIGHT = 0.40
    COLOR_WEIGHT = 0.30

    # Canny edge detection thresholds
    CANNY_LOW = 50
    CANNY_HIGH = 150

    def __init__(
        self,
        min_fps: float = 0.5,
        max_fps: float = 2.0,
        motion_threshold: float = 25.0
    ):
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.motion_threshold = motion_threshold

        logger.debug(
            "Initialized adaptive extractor: fps=%.1f-%.1f, motion_threshold=%.1f",
            min_fps, max_fps, motion_threshold
        )

    def extract_frames(self, video_path: str, chunk: VideoChunk) -> Tuple[np.ndarray, float, float]:
        """
        Extract frames adaptively with complexity calculation.

        Single-pass streaming with motion-based interval adjustment.
        Returns frames, actual fps, and complexity score (0-1).
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(chunk.start_time * video_fps)
        end_frame = int(chunk.end_time * video_fps)

        min_interval = max(1, int(video_fps / self.max_fps))
        max_interval = max(1, int(video_fps / self.min_fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        motion_scores = []
        edge_densities = []
        color_variances = []

        prev_gray = None
        current_idx = start_frame
        next_capture_idx = start_frame

        # Stream through chunk
        while current_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame %d", current_idx)
                break

            if current_idx >= next_capture_idx:
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)

                # Calculate complexity metrics
                edge_densities.append(self._calculate_edge_density(current_gray))
                color_variances.append(np.std(frame))

                if prev_gray is not None:
                    # Calculate motion and adjust interval
                    motion_score = float(np.mean(cv2.absdiff(prev_gray, current_gray)))
                    motion_scores.append(motion_score)

                    interval = min_interval if motion_score > self.motion_threshold else max_interval
                    next_capture_idx += interval
                else:
                    next_capture_idx += min_interval

                prev_gray = current_gray

            current_idx += 1

        cap.release()

        complexity_score = self._calculate_complexity_score(
            motion_scores, edge_densities, color_variances
        )

        actual_fps = len(frames) / chunk.duration if chunk.duration > 0 else 0

        logger.info(
            "Extracted %d frames at %.2f fps, complexity=%.3f",
            len(frames), actual_fps, complexity_score
        )

        return np.array(frames) if frames else np.array([]), actual_fps, complexity_score

    def _calculate_edge_density(self, gray_frame: np.ndarray) -> float:
        """Calculate edge density using Canny detection. Higher = more detail."""
        edges = cv2.Canny(gray_frame, self.CANNY_LOW, self.CANNY_HIGH)
        return float(np.sum(edges > 0) / edges.size)

    def _calculate_complexity_score(
        self,
        motion_scores: list,
        edge_densities: list,
        color_variances: list
    ) -> float:
        """
        Calculate normalized complexity score (0-1).

        Combines motion (30%), edge density (40%), and color variance (30%).
        """
        if not edge_densities:
            return 0.0

        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        avg_edge_density = np.mean(edge_densities)
        avg_color_variance = np.mean(color_variances)

        # Normalize to 0-1
        motion_norm = min(avg_motion / self.MOTION_RANGE, 1.0)
        edge_norm = min(avg_edge_density / self.EDGE_DENSITY_RANGE, 1.0)
        color_norm = min(avg_color_variance / self.COLOR_VARIANCE_RANGE, 1.0)

        complexity = (
            self.MOTION_WEIGHT * motion_norm +
            self.EDGE_WEIGHT * edge_norm +
            self.COLOR_WEIGHT * color_norm
        )

        return float(complexity)

    def extract_single_frame(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Extract single frame at timestamp. Useful for thumbnails."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        cap.release()

        return frame if ret else None
