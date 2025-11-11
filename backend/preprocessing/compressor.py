import logging
import cv2
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Compressor:
    """
    Frame compressor with efficient resizing via pre-allocated arrays.

    Reduces frame dimensions to target resolution, minimizing memory usage
    and improving downstream processing performance.
    """

    DEFAULT_TARGET_WIDTH = 640
    DEFAULT_TARGET_HEIGHT = 480
    DEFAULT_QUALITY = 85

    def __init__(
        self,
        target_width: int = DEFAULT_TARGET_WIDTH,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        quality: int = DEFAULT_QUALITY
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.quality = quality

        logger.debug(
            "Initialized compressor: resolution=%dx%d, quality=%d",
            target_width, target_height, quality
        )

    def compress_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize single frame to target resolution."""
        return cv2.resize(
            frame,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )

    def compress_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize multiple frames efficiently with pre-allocated output array."""
        logger.debug("Compressing %d frames to %dx%d",
                     len(frames), self.target_width, self.target_height)

        n_frames = len(frames)
        if n_frames == 0:
            return np.array([])

        # Pre-allocate output array to avoid list->array conversion overhead
        result = np.zeros(
            (n_frames, self.target_height, self.target_width, frames.shape[3]),
            dtype=frames.dtype
        )

        for i, frame in enumerate(frames):
            result[i] = cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA
            )

        logger.debug("Compression complete: output_shape=%s", result.shape)

        return result

    def get_compression_ratio(self, original_shape: tuple, compressed_shape: tuple) -> float:
        """Calculate compression ratio between original and compressed frame arrays."""
        original_size = np.prod(original_shape)
        compressed_size = np.prod(compressed_shape)

        return original_size / compressed_size if compressed_size > 0 else 1.0
