import logging
import tempfile
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

from models.metadata import VideoChunk, ChunkMetadata
from preprocessing.chunker import Chunker
from preprocessing.frame_extractor import FrameExtractor
from preprocessing.compressor import Compressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Video preprocessing pipeline coordinator.

    Pipeline stages:
        1. Scene-based chunking with duration constraints
        2. Adaptive frame extraction with motion detection
        3. Frame compression to target resolution
        4. Metadata generation with complexity scoring

    Chunks are processed in parallel using ThreadPoolExecutor.
    """

    MAX_WORKERS = 4
    DEFAULT_FPS = 30.0

    def __init__(
        self,
        min_chunk_duration: float = 5.0,
        max_chunk_duration: float = 20.0,
        scene_threshold: float = 13.0,
        min_sampling_fps: float = 0.5,
        max_sampling_fps: float = 2.0,
        motion_threshold: float = 25.0,
        target_width: int = 640,
        target_height: int = 480
    ):
        self.chunker = Chunker(
            min_duration=min_chunk_duration,
            max_duration=max_chunk_duration,
            scene_threshold=scene_threshold
        )
        self.extractor = FrameExtractor(
            min_fps=min_sampling_fps,
            max_fps=max_sampling_fps,
            motion_threshold=motion_threshold
        )
        self.compressor = Compressor(
            target_width=target_width,
            target_height=target_height
        )

        self._video_metadata_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Preprocessor initialized: chunks=%.1f-%.1fs, scene_threshold=%.1f, "
            "fps=%.1f-%.1f, motion_threshold=%.1f, resolution=%dx%d",
            min_chunk_duration, max_chunk_duration, scene_threshold,
            min_sampling_fps, max_sampling_fps, motion_threshold,
            target_width, target_height
        )

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata with caching to avoid redundant file opens."""
        if video_path in self._video_metadata_cache:
            return self._video_metadata_cache[video_path]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Validate and apply fallbacks
        if fps <= 0 or fps != fps:  # Check for invalid/NaN
            logger.warning("Invalid FPS %.2f for %s, using default %.1f", fps, video_path, self.DEFAULT_FPS)
            fps = self.DEFAULT_FPS

        if frame_count <= 0:
            logger.warning("Invalid frame count %d for %s", frame_count, video_path)
            frame_count = 0

        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0

        metadata = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }
        self._video_metadata_cache[video_path] = metadata

        logger.debug("Cached metadata: fps=%.2f, frames=%d, duration=%.2fs", fps, frame_count, duration)
        return metadata

    def process_video_from_bytes(
        self,
        video_bytes: bytes,
        video_id: str,
        filename: str,
        hashed_identifier: str = ""
    ) -> List[Dict[str, Any]]:
        """Process video from uploaded bytes with automatic temp file cleanup."""
        logger.info("Starting preprocessing: video_id=%s, filename=%s", video_id, filename)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()

            try:
                return self.process_video(
                    video_path=temp_file.name,
                    video_id=video_id,
                    filename=filename,
                    hashed_identifier=hashed_identifier
                )
            except Exception as e:
                logger.error("Processing failed for video_id=%s: %s", video_id, e)
                raise

    def process_video(
        self,
        video_path: str,
        video_id: str,
        filename: str,
        hashed_identifier: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Run complete preprocessing pipeline with parallel chunk processing.

        Returns list of processed chunks with frames, metadata, and complexity scores.
        """
        logger.info("Processing video: video_id=%s, path=%s", video_id, video_path)

        # Cache metadata once for reuse
        self._get_video_metadata(video_path)

        # Stage 1: Chunk video
        logger.info("Stage 1/4: Chunking video")
        chunks = self.chunker.chunk_video(video_path, video_id)
        logger.info("Created %d chunks", len(chunks))

        if not chunks:
            logger.warning("No chunks created for video_id=%s", video_id)
            return []

        # Stage 2-4: Process chunks in parallel
        max_workers = min(self.MAX_WORKERS, len(chunks))
        logger.info("Processing %d chunks with %d workers", len(chunks), max_workers)

        processed_chunks = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self._process_single_chunk,
                    chunk, video_path, video_id, filename, hashed_identifier
                ): chunk
                for chunk in chunks
            }

            for i, future in enumerate(as_completed(future_to_chunk), 1):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        processed_chunks.append(result)
                        logger.info(
                            "Chunk %d/%d: %s (%d frames, %.2fMB, complexity=%.3f)",
                            i, len(chunks), result['chunk_id'],
                            result['metadata']['frame_count'],
                            result['memory_mb'],
                            result['metadata']['complexity_score']
                        )
                except Exception as e:
                    logger.error("Chunk %s failed: %s", chunk.chunk_id, e)

        total_frames = sum(c['metadata']['frame_count'] for c in processed_chunks)
        logger.info("Preprocessing complete: %d/%d chunks, %d total frames",
                    len(processed_chunks), len(chunks), total_frames)
        return processed_chunks

    def _process_single_chunk(
        self,
        chunk: VideoChunk,
        video_path: str,
        video_id: str,
        filename: str,
        hashed_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process single chunk: extract → compress → package.

        Thread-safe for parallel execution.
        Returns None if processing fails.
        """
        #TODO: Specifcy explicit return type and not just a dict in docstring
        try:
            # Extract frames with complexity analysis
            frames, sampling_fps, complexity_score = self.extractor.extract_frames(video_path, chunk)

            if len(frames) == 0:
                logger.warning("No frames extracted for %s, skipping", chunk.chunk_id)
                return None

            # Compress frames
            compressed_frames = self.compressor.compress_frames(frames)

            # Calculate sizes
            original_mb = frames.nbytes / (1024 * 1024)
            compressed_mb = compressed_frames.nbytes / (1024 * 1024)
            ratio = self.compressor.get_compression_ratio(frames.shape, compressed_frames.shape)

            logger.debug(
                "%s: compressed %.2fMB → %.2fMB (%.1fx)",
                chunk.chunk_id, original_mb, compressed_mb, ratio
            )

            # Create metadata
            metadata = self._create_metadata(
                chunk=chunk,
                video_id=video_id,
                filename=filename,
                hashed_identifier=hashed_identifier,
                frame_count=len(compressed_frames),
                sampling_fps=sampling_fps,
                complexity_score=complexity_score
            )

            #TODO: Turn this into a dataclass or specific type
            return {
                'chunk_id': chunk.chunk_id,
                'frames': compressed_frames,
                'metadata': metadata.to_dict(),
                'memory_mb': compressed_mb
            }

        except Exception as e:
            logger.error("Error processing %s: %s", chunk.chunk_id, e)
            return None

    def _create_metadata(
        self,
        chunk: VideoChunk,
        video_id: str,
        filename: str,
        hashed_identifier: str,
        frame_count: int,
        sampling_fps: float,
        complexity_score: float
    ) -> ChunkMetadata:
        """Create metadata object from chunk processing results."""
        file_type = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'mp4'

        return ChunkMetadata(
            chunk_id=chunk.chunk_id,
            video_id=video_id,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            duration=chunk.duration,
            frame_count=frame_count,
            sampling_fps=sampling_fps,
            complexity_score=complexity_score,
            original_filename=filename,
            file_type=file_type,
            hashed_identifier=hashed_identifier
        )

    def get_stats(self, processed_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # TODO: Improve docstring and be explicit about parameter and return types
        """Calculate aggregate statistics from processed chunks."""
        if not processed_chunks:
            # TODO: Turn this into a dataclass
            return {
                'total_chunks': 0,
                'total_frames': 0,
                'total_duration': 0,
                'total_memory_mb': 0,
                'avg_chunk_duration': 0,
                'avg_frames_per_chunk': 0,
                'avg_complexity': 0,
                'avg_sampling_fps': 0,
                'complexity_range': (0, 0)
            }

        total_frames = sum(c['metadata']['frame_count'] for c in processed_chunks)
        total_duration = sum(c['metadata']['duration'] for c in processed_chunks)
        total_memory = sum(c['memory_mb'] for c in processed_chunks)

        complexities = [c['metadata']['complexity_score'] for c in processed_chunks]
        fps_values = [c['metadata']['sampling_fps'] for c in processed_chunks]

        # TODO: Turn this into a dataclass
        return {
            'total_chunks': len(processed_chunks),
            'total_frames': total_frames,
            'total_duration': total_duration,
            'total_memory_mb': total_memory,
            'avg_chunk_duration': total_duration / len(processed_chunks),
            'avg_frames_per_chunk': total_frames / len(processed_chunks),
            'avg_complexity': float(np.mean(complexities)),
            'avg_sampling_fps': float(np.mean(fps_values)),
            'complexity_range': (float(min(complexities)), float(max(complexities)))
        }
