import logging
import tempfile
from typing import List, Dict, Any
from pathlib import Path
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
    1. Chunk video using fixed intervals
    2. Extract frames at fixed rate per chunk
    3. Compress frames to target resolution
    4. Package with metadata
    """
    
    def __init__(
        self,
        chunk_duration: float = 10.0,
        sampling_fps: float = 1.0,
        target_width: int = 640,
        target_height: int = 480
    ):
        """
        Initialize all components.
        
        Args:
            chunk_duration: Target duration for each chunk in seconds
            sampling_fps: Frames per second to extract
            target_width: Target width for compressed frames
            target_height: Target height for compressed frames
        """
        self.chunker = Chunker(chunk_duration=chunk_duration)
        self.extractor = FrameExtractor(sampling_fps=sampling_fps)
        self.compressor = Compressor(
            target_width=target_width,
            target_height=target_height
        )
        
        logger.info(
            "Preprocessor initialized: chunk_duration=%ss, sampling_fps=%s, resolution=%sx%s",
            chunk_duration, sampling_fps, target_width, target_height
        )
    
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
        
        # Write bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name
        
        try:
            # Process the video
            result = self.process_video(
                video_path=temp_path,
                video_id=video_id,
                filename=filename,
                s3_url=s3_url
            )
            return result
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
    
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
        
        # Step 1: Chunk the video
        logger.info("Step 1/4: Chunking video")
        chunks = self.chunker.chunk_video(video_path, video_id)
        logger.info("Created %d chunks", len(chunks))
        
        # Step 2-4: Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks, 1):
            logger.info("Processing chunk %d/%d: %s", i, len(chunks), chunk.chunk_id)
            
            # Step 2: Extract frames
            frames, sampling_fps = self.extractor.extract_frames(video_path, chunk)
            
            if len(frames) == 0:
                logger.warning("No frames extracted for chunk %s, skipping", chunk.chunk_id)
                continue
            
            logger.debug("Extracted %d frames at %.2f fps", len(frames), sampling_fps)
            
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
                complexity_score=0.5  # TODO: Add complexity calculation
            )
            
            # Package chunk data
            chunk_data = {
                'chunk_id': chunk.chunk_id,
                'frames': compressed_frames,  # numpy array
                'metadata': metadata.to_dict(),
                'memory_mb': compressed_size
            }
            processed_chunks.append(chunk_data)
            
            logger.info(
                "Chunk processed: %s (%d frames, %.2fMB)",
                chunk.chunk_id, len(compressed_frames), compressed_size
            )
        
        logger.info("Preprocessing complete: %d chunks, %d total frames",
                    len(processed_chunks),
                    sum(len(c['frames']) for c in processed_chunks))
        return processed_chunks
    
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
        
    
