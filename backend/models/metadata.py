from dataclasses import dataclass, field
from typing import Dict
from datetime import datetime
import numpy as np


@dataclass
class VideoChunk:
    """
    Represents a single chunk of video with timing information.
    This is the core unit we'll process.
    """
    chunk_id: str          # Unique ID like "video_123_chunk_0001"
    start_time: float      # Start time in seconds
    end_time: float        # End time in seconds
    
    @property
    def duration(self) -> float:
        """Calculate chunk duration in seconds"""
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"VideoChunk(id={self.chunk_id}, {self.start_time:.1f}-{self.end_time:.1f}s)"


@dataclass
class ChunkMetadata:
    """
    Complete metadata for a processed chunk.
    This will be stored in the database alongside embeddings.
    """
    # Identifiers
    chunk_id: str
    video_id: str
    
    # Timing
    start_time: float
    end_time: float
    duration: float
    
    # Processing info
    frame_count: int
    sampling_fps: float
    complexity_score: float
    
    # File info
    original_filename: str
    file_type: str
    hashed_identifier: str
    
    # Timestamps
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'chunk_id': self.chunk_id,
            'video_id': self.video_id,
            'timestamp_range': [self.start_time, self.end_time],
            'duration': self.duration,
            'frame_count': self.frame_count,
            'sampling_fps': self.sampling_fps,
            'complexity_score': self.complexity_score,
            'file_info': {
                'filename': self.original_filename,
                'type': self.file_type,
                'hashed_identifier': self.hashed_identifier
            },
            'processed_at': self.processing_timestamp.isoformat()
        }


@dataclass
class ProcessedChunk:
    """
    Container for a fully processed chunk ready for embedding generation.
    """
    chunk_id: str
    frames: np.ndarray      # Shape: (n_frames, height, width, 3) - frames stored in BGR order (cv2 default)
    metadata: ChunkMetadata
    
    def __repr__(self):
        return f"ProcessedChunk(id={self.chunk_id}, frames={self.frames.shape})"
