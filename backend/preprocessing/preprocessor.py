import logging
import tempfile
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

from models.metadata import VideoChunk, ChunkMetadata, ProcessedChunk
from preprocessing.chunker import Chunker
from preprocessing.frame_extractor import FrameExtractor
from preprocessing.compressor import Compressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Main preprocessing pipeline that coordinates all components.

    Flow:
    1. Chunk video using scene detection with duration constraints
    2. Extract frames adaptively based on motion/complexity
    3. Compress frames to target resolution
    4. Package with metadata
    """

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
        """
        Initialize all components.

        Args:
            min_chunk_duration: Minimum chunk duration in seconds
            max_chunk_duration: Maximum chunk duration in seconds
            scene_threshold: Scene detection sensitivity (lower = more sensitive)
            min_sampling_fps: Minimum frames per second for static scenes
            max_sampling_fps: Maximum frames per second for high-motion scenes
            motion_threshold: Motion detection threshold (higher = less sensitive)
            target_width: Target width for compressed frames
            target_height: Target height for compressed frames
        """
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

        # Video metadata cache to avoid re-opening files
        self._video_metadata_cache = {}

        logger.info(
            "Preprocessor initialized: chunk_range=%.1fs-%.1fs, scene_threshold=%.1f, adaptive_fps=%.1f-%.1f, motion_threshold=%.1f, resolution=%dx%d",
            min_chunk_duration, max_chunk_duration, scene_threshold, min_sampling_fps, max_sampling_fps, motion_threshold, target_width, target_height
        )

    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Get video metadata with caching to avoid re-opening files.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with fps, frame_count, duration
        """
        if video_path not in self._video_metadata_cache:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Validate metadata
            if fps <= 0 or fps != fps:  # Check for 0, negative, NaN
                logger.warning("Invalid FPS %.2f for %s, using default 30.0", fps, video_path)
                fps = 30.0

            if frame_count <= 0:
                logger.warning("Invalid frame count %d for %s", frame_count, video_path)
                frame_count = 0

            duration = frame_count / fps if fps > 0 and frame_count > 0 else 0

            self._video_metadata_cache[video_path] = {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration
            }
            logger.debug("Cached metadata for %s: fps=%.2f, frames=%d, duration=%.2fs",
                        video_path, fps, frame_count, duration)

        return self._video_metadata_cache[video_path]

    def process_video_from_bytes(
        self,
        video_bytes: bytes,
        video_id: str,
        filename: str,
        s3_url: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Process video from bytes (uploaded file).
        
        Args:
            video_bytes: Video file as bytes
            video_id: Unique identifier for video
            filename: Original filename
            s3_url: S3 URL (optional)
            
        Returns:
            List of processed chunk dictionaries
        """
        logger.info("Starting preprocessing: video_id=%s, filename=%s", video_id, filename)

        # Write bytes to temporary file with auto-cleanup
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()  # Ensure bytes are written to disk

            # Process the video (temp file auto-deleted when context exits)
            try:
                result = self.process_video(
                    video_path=temp_file.name,
                    video_id=video_id,
                    filename=filename,
                    s3_url=s3_url
                )
                return result
            except Exception as e:
                logger.error("Processing failed for video_id=%s: %s", video_id, e)
                raise
    
    def process_video(
        self,
        video_path: str,
        video_id: str,
        filename: str,
        s3_url: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            video_path: Local path to video file
            video_id: Unique identifier for video
            filename: Original filename
            s3_url: S3 URL for metadata
            
        Returns:
            List of processed chunk dictionaries
        """
        logger.info("Processing video: video_id=%s, path=%s", video_id, video_path)

        # Cache video metadata to avoid re-opening file
        self._get_video_metadata(video_path)

        # Step 1: Chunk the video
        logger.info("Step 1/4: Chunking video")
        chunks = self.chunker.chunk_video(video_path, video_id)
        logger.info("Created %d chunks", len(chunks))

        # Step 2-4: Process chunks in parallel
        processed_chunks = []

        # Use ThreadPoolExecutor for parallel I/O-bound work
        max_workers = min(4, len(chunks))  # Don't create more threads than chunks
        logger.info("Processing chunks with %d parallel workers", max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._process_single_chunk,
                    chunk, video_path, video_id, filename, s3_url
                ): chunk
                for chunk in chunks
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_chunk), 1):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        processed_chunks.append(result)
                        logger.info(
                            "Chunk %d/%d completed: %s (%d frames, %.2fMB)",
                            i, len(chunks), result['chunk_id'],
                            result['metadata']['frame_count'],
                            result['memory_mb']
                        )
                except Exception as e:
                    logger.error("Failed to process chunk %s: %s", chunk.chunk_id, e)

        logger.info("Preprocessing complete: %d chunks, %d total frames",
                    len(processed_chunks),
                    sum(c['metadata']['frame_count'] for c in processed_chunks))
        return processed_chunks

    def _process_single_chunk(
        self,
        chunk: VideoChunk,
        video_path: str,
        video_id: str,
        filename: str,
        s3_url: str
    ) -> Dict[str, Any]:
        """
        Process a single chunk (extract, compress, package).
        Can be called from thread pool for parallel processing.

        Args:
            chunk: Video chunk to process
            video_path: Path to video file
            video_id: Video identifier
            filename: Original filename
            s3_url: S3 URL

        Returns:
            Processed chunk dictionary or None if failed
        """
        try:
            logger.debug("Processing chunk: %s", chunk.chunk_id)

            # Step 2: Extract frames with complexity calculation
            frames, sampling_fps, complexity_score = self.extractor.extract_frames(video_path, chunk)

            if len(frames) == 0:
                logger.warning("No frames extracted for chunk %s, skipping", chunk.chunk_id)
                return None

            logger.debug("Extracted %d frames at %.2f fps, complexity=%.3f", len(frames), sampling_fps, complexity_score)

            # Step 3: Compress frames
            compressed_frames = self.compressor.compress_frames(frames)

            # Calculate compression stats
            original_size = frames.nbytes / (1024 * 1024)  # MB
            compressed_size = compressed_frames.nbytes / (1024 * 1024)  # MB
            compression_ratio = self.compressor.get_compression_ratio(
                frames.shape, compressed_frames.shape
            )

            logger.debug(
                "Compressed frames: %.2fMB -> %.2fMB (%.2fx reduction)",
                original_size, compressed_size, compression_ratio
            )

            # Step 4: Create metadata
            metadata = self._create_metadata(
                chunk=chunk,
                video_id=video_id,
                filename=filename,
                s3_url=s3_url,
                frame_count=len(compressed_frames),
                sampling_fps=sampling_fps,
                complexity_score=complexity_score
            )

            # Package chunk data
            return {
                'chunk_id': chunk.chunk_id,
                'frames': compressed_frames,  # numpy array
                'metadata': metadata.to_dict(),
                'memory_mb': compressed_size
            }

        except Exception as e:
            logger.error("Error processing chunk %s: %s", chunk.chunk_id, e)
            return None
    
    def _create_metadata(
        self,
        chunk: VideoChunk,
        video_id: str,
        filename: str,
        s3_url: str,
        frame_count: int,
        sampling_fps: float,
        complexity_score: float
    ) -> ChunkMetadata:
        """Create complete metadata object"""
        
        # Extract file type from filename
        file_type = filename.split('.')[-1].lower() if '.' in filename else 'mp4'
        
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
            original_s3_url=s3_url
        )
    
    def get_stats(self, processed_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics about the preprocessing.
        Useful for monitoring and optimization.
        """
        total_frames = sum(chunk['metadata']['frame_count'] for chunk in processed_chunks)
        total_duration = sum(chunk['metadata']['duration'] for chunk in processed_chunks)
        total_memory = sum(chunk['memory_mb'] for chunk in processed_chunks)
        
        complexities = [chunk['metadata']['complexity_score'] for chunk in processed_chunks]
        fps_values = [chunk['metadata']['sampling_fps'] for chunk in processed_chunks]
        
        return {
            'total_chunks': len(processed_chunks),
            'total_frames': total_frames,
            'total_duration': total_duration,
            'total_memory_mb': total_memory,
            'avg_chunk_duration': total_duration / len(processed_chunks) if processed_chunks else 0,
            'avg_frames_per_chunk': total_frames / len(processed_chunks) if processed_chunks else 0,
            'avg_complexity': np.mean(complexities) if complexities else 0,
            'avg_sampling_fps': np.mean(fps_values) if fps_values else 0,
            'complexity_range': (min(complexities), max(complexities)) if complexities else (0, 0)
        }
        
    
