from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import numpy as np


@dataclass
class ChunkMetadata:
    """
    Comprehensive metadata for a single video chunk.
    
    This metadata is consistent across all modalities (video, audio, embeddings)
    and enables precise seeking, filtering, and debugging.
    """
    
    # Core identifiers
    chunk_id: str  # Format: {video_id}_chunk_{0000}
    video_id: str  # Original video identifier
    
    # Temporal information
    timestamp_range: tuple[float, float]  # (start_seconds, end_seconds)
    duration: float  # Duration in seconds
    
    # Frame extraction details
    frame_count: int  # Number of frames extracted from this chunk
    sampling_fps: float  # Actual frames per second used for this chunk
    complexity_score: float  # 0.0-1.0, determines adaptive sampling rate
    
    # Frame indices (for reconstruction and debugging)
    frame_indices: List[int] = field(default_factory=list)  # Original frame numbers from video
    filtered_indices: List[int] = field(default_factory=list)  # Frames removed by quality filter
    
    # Source video information
    original_s3_url: str = ""  # S3 URL of original video
    file_type: str = ""  # Video container format (mp4, mov, etc)
    resolution: tuple[int, int] = (0, 0)  # (width, height) in pixels
    
    # Processing metadata
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    scene_boundary: bool = False  # True if chunk starts at scene boundary
    
    def __post_init__(self):
        """Validate metadata after initialization"""
        # Ensure timestamp range is valid
        if self.timestamp_range[0] >= self.timestamp_range[1]:
            raise ValueError(f"Invalid timestamp range: {self.timestamp_range}")
        
        # Ensure duration matches timestamp range
        expected_duration = self.timestamp_range[1] - self.timestamp_range[0]
        if abs(self.duration - expected_duration) > 0.1:  # 100ms tolerance
            raise ValueError(
                f"Duration mismatch: {self.duration}s vs timestamp range "
                f"{expected_duration}s"
            )
        
        # Validate chunk duration constraints (5-20 seconds)
        if not (5.0 <= self.duration <= 20.0):
            raise ValueError(
                f"Chunk duration {self.duration}s outside valid range [5.0, 20.0]"
            )
        
        # Validate complexity score
        if not (0.0 <= self.complexity_score <= 1.0):
            raise ValueError(f"Complexity score {self.complexity_score} must be in [0.0, 1.0]")
        
        # Validate sampling rate
        if not (0.5 <= self.sampling_fps <= 2.0):
            raise ValueError(f"Sampling FPS {self.sampling_fps} must be in [0.5, 2.0]")
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON serialization"""
        return {
            'chunk_id': self.chunk_id,
            'video_id': self.video_id,
            'timestamp_range': list(self.timestamp_range),
            'duration': self.duration,
            'frame_count': self.frame_count,
            'sampling_fps': self.sampling_fps,
            'complexity_score': self.complexity_score,
            'frame_indices': self.frame_indices,
            'filtered_indices': self.filtered_indices,
            'original_s3_url': self.original_s3_url,
            'file_type': self.file_type,
            'resolution': list(self.resolution),
            'processing_timestamp': self.processing_timestamp.isoformat(),
            'scene_boundary': self.scene_boundary
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChunkMetadata':
        """Create metadata from dictionary"""
        # Convert timestamp string back to datetime
        if isinstance(data.get('processing_timestamp'), str):
            data['processing_timestamp'] = datetime.fromisoformat(
                data['processing_timestamp']
            )
        
        # Convert lists back to tuples where needed
        if isinstance(data.get('timestamp_range'), list):
            data['timestamp_range'] = tuple(data['timestamp_range'])
        if isinstance(data.get('resolution'), list):
            data['resolution'] = tuple(data['resolution'])
        
        return cls(**data)


@dataclass
class ProcessedChunk:
    """
    Complete processed chunk ready for embedding.
    
    Combines extracted frames with comprehensive metadata for downstream processing.
    """
    
    # Extracted frames
    frames: np.ndarray  # Shape: (n_frames, height, width, 3), dtype: uint8
    
    # Comprehensive metadata
    metadata: ChunkMetadata
    
    def __post_init__(self):
        """Validate processed chunk after initialization"""
        # Validate frames array shape
        if len(self.frames.shape) != 4:
            raise ValueError(
                f"Frames must be 4D array (n_frames, height, width, channels), "
                f"got shape {self.frames.shape}"
            )
        
        if self.frames.shape[3] != 3:
            raise ValueError(
                f"Frames must have 3 color channels (RGB), got {self.frames.shape[3]}"
            )
        
        # Validate frame count matches metadata
        if self.frames.shape[0] != self.metadata.frame_count:
            raise ValueError(
                f"Frame count mismatch: array has {self.frames.shape[0]} frames, "
                f"metadata says {self.metadata.frame_count}"
            )
        
        # Validate data type
        if self.frames.dtype != np.uint8:
            raise ValueError(f"Frames must be uint8, got {self.frames.dtype}")
    
    def get_thumbnail(self) -> np.ndarray:
        """Get first frame as thumbnail"""
        return self.frames[0]
    
    def get_memory_size_mb(self) -> float:
        """Calculate approximate memory size in megabytes"""
        # Frames array size
        frames_size = self.frames.nbytes
        
        # Metadata size (approximate)
        metadata_size = 1024  # ~1KB for metadata
        
        total_bytes = frames_size + metadata_size
        return total_bytes / (1024 * 1024)
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.
        
        Note: Frames are NOT included in dictionary form to avoid
        unnecessary serialization overhead. Use numpy save/load for frames.
        """
        return {
            'metadata': self.metadata.to_dict(),
            'frame_shape': self.frames.shape,
            'memory_size_mb': self.get_memory_size_mb()
        }


# Utility functions for chunk ID generation
def generate_chunk_id(video_id: str, chunk_index: int) -> str:
    """
    Generate consistent chunk ID for multimodal alignment.
    
    Format: {video_id}_chunk_{0000}
    Supports up to 9999 chunks per video (166 hours @ 10s/chunk)
    
    Args:
        video_id: Unique identifier for the video
        chunk_index: Zero-based chunk index
    
    Returns:
        Formatted chunk ID string
    """
    if chunk_index < 0 or chunk_index > 9999:
        raise ValueError(f"Chunk index {chunk_index} must be in range [0, 9999]")
    
    return f"{video_id}_chunk_{chunk_index:04d}"


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """
    Parse chunk ID to extract video ID and chunk index.
    
    Args:
        chunk_id: Chunk ID string in format {video_id}_chunk_{0000}
    
    Returns:
        Tuple of (video_id, chunk_index)
    """
    parts = chunk_id.rsplit('_chunk_', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid chunk ID format: {chunk_id}")
    
    video_id = parts[0]
    try:
        chunk_index = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid chunk index in chunk ID: {chunk_id}")
    
    return video_id, chunk_index
