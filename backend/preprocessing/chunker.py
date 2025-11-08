import logging
from typing import List, Tuple
import cv2
from scenedetect import detect, ContentDetector

from models.metadata import VideoChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Chunker:
    """
    Scene-based video chunker with adaptive duration constraints.

    Detects scene boundaries, merges short scenes, splits long scenes, and creates
    chunks that respect semantic boundaries while meeting duration requirements.
    """

    DEFAULT_MIN_DURATION = 5.0
    DEFAULT_MAX_DURATION = 20.0
    DEFAULT_SCENE_THRESHOLD = 13.0

    def __init__(
        self,
        min_duration: float = DEFAULT_MIN_DURATION,
        max_duration: float = DEFAULT_MAX_DURATION,
        scene_threshold: float = DEFAULT_SCENE_THRESHOLD
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.scene_threshold = scene_threshold

        logger.debug(
            "Initialized chunker: duration=%.1f-%.1fs, scene_threshold=%.1f",
            min_duration, max_duration, scene_threshold
        )
    
    def chunk_video(self, video_path: str, video_id: str) -> List[VideoChunk]:
        """Detect scenes, apply duration constraints, and create VideoChunk objects."""
        logger.info("Chunking video: path=%s, video_id=%s, threshold=%.1f",
                    video_path, video_id, self.scene_threshold)

        try:
            scenes = detect(video_path, ContentDetector(threshold=self.scene_threshold))
            logger.info("Detected %d raw scene boundaries", len(scenes))
        except Exception as e:
            logger.error("Scene detection failed: %s, falling back to fixed chunking", e)
            return self._fallback_chunking(video_path, video_id)

        if not scenes:
            logger.warning("No scenes detected, using fallback chunking")
            return self._fallback_chunking(video_path, video_id)

        constrained_scenes = self._apply_duration_constraints(scenes)
        logger.info("After constraints: %d chunks (min=%.1fs, max=%.1fs)",
                    len(constrained_scenes), self.min_duration, self.max_duration)

        chunks = []
        for idx, (start_time, end_time) in enumerate(constrained_scenes):
            chunk_id = f"{video_id}_chunk_{idx:04d}"
            chunk = VideoChunk(
                chunk_id=chunk_id,
                start_time=start_time,
                end_time=end_time
            )
            chunks.append(chunk)
            logger.debug("Created chunk: %s (%.1fs-%.1fs, duration=%.1fs)",
                        chunk_id, start_time, end_time, chunk.duration)

        return chunks
    
    def _apply_duration_constraints(
        self,
        scenes: List[Tuple]
    ) -> List[Tuple[float, float]]:
        """Merge scenes shorter than min_duration, split scenes longer than max_duration."""
        result = []
        i = 0

        while i < len(scenes):
            start_tc, end_tc = scenes[i]
            start_time = start_tc.get_seconds()
            end_time = end_tc.get_seconds()
            duration = end_time - start_time

            # Merge short scenes with next
            while duration < self.min_duration and i + 1 < len(scenes):
                logger.debug("Merging short scene at index %d (%.1fs) with next", i, duration)
                i += 1
                end_tc = scenes[i][1]
                end_time = end_tc.get_seconds()
                duration = end_time - start_time

            # Split long scenes into equal parts
            if duration > self.max_duration:
                n_parts = int(duration / self.max_duration) + 1
                part_duration = duration / n_parts
                logger.debug("Splitting long scene (%.1fs) into %d parts of ~%.1fs each",
                            duration, n_parts, part_duration)

                for j in range(n_parts):
                    part_start = start_time + j * part_duration
                    part_end = min(part_start + part_duration, end_time)
                    result.append((part_start, part_end))
            else:
                result.append((start_time, end_time))

            i += 1

        return result
    
    def _fallback_chunking(self, video_path: str, video_id: str) -> List[VideoChunk]:
        """Create fixed-duration chunks when scene detection fails."""
        logger.warning("Using fallback fixed-duration chunking")

        chunk_duration = (self.min_duration + self.max_duration) / 2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        logger.info("Fallback: video_duration=%.1fs, chunk_duration=%.1fs",
                    duration, chunk_duration)

        chunks = []
        chunk_start = 0.0
        chunk_idx = 0

        while chunk_start < duration:
            chunk_end = min(chunk_start + chunk_duration, duration)

            chunk_id = f"{video_id}_chunk_{chunk_idx:04d}"
            chunk = VideoChunk(
                chunk_id=chunk_id,
                start_time=chunk_start,
                end_time=chunk_end
            )
            chunks.append(chunk)

            logger.debug("Created chunk: %s (%.1fs-%.1fs)", chunk_id, chunk_start, chunk_end)

            chunk_start = chunk_end
            chunk_idx += 1

        logger.info("Fallback created %d chunks", len(chunks))
        return chunks

