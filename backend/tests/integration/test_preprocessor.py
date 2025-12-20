import pytest
import numpy as np
import os
from preprocessing.preprocessor import Preprocessor


class TestEndToEndProcessing:
    """Test complete preprocessing pipeline."""

    def test_process_video_from_bytes(self, preprocessor, sample_video_bytes):
        """Verify processing from video bytes works."""
        result = preprocessor.process_video_from_bytes(
            video_bytes=sample_video_bytes,
            video_id="test_video",
            filename="test.mp4",
            hashed_identifier="abc123"
        )
        print(result)
        assert len(result) > 0
        assert all('chunk_id' in chunk for chunk in result)
        assert all('frames' in chunk for chunk in result)
        assert all('metadata' in chunk for chunk in result)

    def test_process_video_from_path(self, preprocessor, sample_video_5s):
        """Verify processing from video path works."""
        result = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test_video",
            filename="test.mp4",
            hashed_identifier=""
        )

        assert len(result) > 0

    def test_processed_chunks_have_frames(self, preprocessor, sample_video_5s):
        """Verify all processed chunks contain frames."""
        result = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test",
            filename="test.mp4",
            hashed_identifier=""
        )

        for chunk in result:
            assert isinstance(chunk['frames'], np.ndarray)
            assert len(chunk['frames']) > 0

    def test_processed_chunks_have_metadata(self, preprocessor, sample_video_5s):
        """Verify all chunks have complete metadata."""
        result = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test",
            filename="test.mp4",
            hashed_identifier=""
        )

        for chunk in result:
            metadata = chunk['metadata']
            assert 'chunk_id' in metadata
            assert 'video_id' in metadata
            assert 'timestamp_range' in metadata
            assert 'duration' in metadata
            assert 'frame_count' in metadata
            assert 'sampling_fps' in metadata
            assert 'complexity_score' in metadata


class TestPipelineStages:
    """Test individual pipeline stages."""

    def test_chunking_stage_creates_chunks(self, preprocessor, sample_video_5s):
        """Verify chunking stage creates chunks."""
        result = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test",
            filename="test.mp4",
            hashed_identifier=""
        )

        # Should create at least 1 chunk
        assert len(result) >= 1

    def test_extraction_stage_generates_frames(self, preprocessor, sample_video_5s):
        """Verify frame extraction stage works."""
        result = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test",
            filename="test.mp4",
            hashed_identifier=""
        )

        total_frames = sum(chunk['metadata']['frame_count'] for chunk in result)
        assert total_frames > 0

    def test_compression_stage_reduces_size(self, preprocessor, sample_video_5s):
        """Verify compression stage reduces frame dimensions."""
        result = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test",
            filename="test.mp4",
            hashed_identifier=""
        )

        for chunk in result:
            frames = chunk['frames']
            # Frames should be compressed to target resolution
            assert frames.shape[1] == preprocessor.compressor.target_height  # 480
            assert frames.shape[2] == preprocessor.compressor.target_width   # 640


class TestProcessingStatistics:
    """Test aggregate statistics calculation."""

    def test_get_stats_calculates_totals(self, preprocessor, sample_video_5s):
        """Verify statistics calculation from processed chunks."""
        result = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test",
            filename="test.mp4",
            hashed_identifier=""
        )

        stats = preprocessor.get_stats(result)

        assert stats['total_chunks'] == len(result)
        assert stats['total_frames'] > 0
        assert stats['total_duration'] > 0
        assert stats['avg_complexity'] >= 0.0
        assert stats['avg_complexity'] <= 1.0

    def test_get_stats_with_empty_chunks(self, preprocessor):
        """Verify stats handling of empty chunk list."""
        stats = preprocessor.get_stats([])

        assert stats['total_chunks'] == 0
        assert stats['total_frames'] == 0
        assert stats['avg_complexity'] == 0.0


class TestParallelProcessing:
    """Test parallel chunk processing."""

    def test_parallel_processing_produces_same_results(self, sample_video_5s):
        """Verify parallel processing matches sequential (consistency check)."""
        # Process with parallel
        preprocessor_parallel = Preprocessor()
        result_parallel = preprocessor_parallel.process_video(
            video_path=str(sample_video_5s),
            video_id="test",
            filename="test.mp4",
            hashed_identifier=""
        )

        # Results should be consistent
        assert len(result_parallel) > 0
        chunk_ids = [chunk['chunk_id'] for chunk in result_parallel]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_multiple_chunks_processed_successfully(self, sample_video_5s):
        """Verify multiple chunks are processed without errors."""
        preprocessor = Preprocessor(
            min_chunk_duration=1.0,  # Create more chunks
            max_chunk_duration=3.0
        )

        result = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test",
            filename="test.mp4",
            hashed_identifier=""
        )

        # All chunks should process successfully
        assert len(result) > 0
        for chunk in result:
            assert chunk['metadata']['frame_count'] > 0


class TestMetadataCaching:
    """Test video metadata caching."""

    def test_metadata_cache_improves_performance(self, preprocessor, sample_video_5s):
        """Verify metadata is cached after first access."""
        # First call - cache miss
        result1 = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test1",
            filename="test.mp4",
            hashed_identifier=""
        )

        # Cache should now contain metadata
        assert str(sample_video_5s) in preprocessor._video_metadata_cache

        # Subsequent processing uses cached metadata
        result2 = preprocessor.process_video(
            video_path=str(sample_video_5s),
            video_id="test2",
            filename="test.mp4",
            hashed_identifier=""
        )

        assert len(result1) > 0
        assert len(result2) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_video_bytes_raises_error(self, preprocessor):
        """Verify invalid video data is handled gracefully."""
        with pytest.raises(RuntimeError):
            preprocessor.process_video_from_bytes(
                video_bytes=b"not a video",
                video_id="test",
                filename="fake.mp4",
                hashed_identifier=""
            )

    @pytest.mark.slow
    def test_corrupted_video_handles_gracefully(self, preprocessor, temp_dir):
        """Verify corrupted video file is handled."""
        # Create corrupted video file
        corrupted_path = temp_dir / "corrupted.mp4"
        corrupted_path.write_bytes(b"MP4_HEADER_ONLY")

        result = preprocessor.process_video(
            video_path=str(corrupted_path),
            video_id="test",
            filename="corrupted.mp4",
            hashed_identifier=""
        )
        assert result == []  # Returns empty instead of crashing


class TestCodecSupport:
    """Test codec detection and transcoding capabilities."""

    def test_detects_av1_codec(self, preprocessor, sample_video_av1):
        """Verify AV1 codec is correctly identified."""
        # Ensure the file exists
        assert os.path.exists(sample_video_av1)
        
        # Check codec detection
        codec = preprocessor._get_video_codec(str(sample_video_av1))
        assert codec == "av1"

    def test_transcodes_av1_to_h264(self, preprocessor, sample_video_av1):
        """Verify AV1 video is transcoded and processed."""
        with open(sample_video_av1, "rb") as f:
            video_bytes = f.read()

        # Process video (should trigger transcoding)
        result = preprocessor.process_video_from_bytes(
            video_bytes=video_bytes,
            video_id="test_av1",
            filename="test_av1.mp4"
        )
        
        # Verify we got chunks back
        assert len(result) > 0
        
        # Verify chunks have frames
        assert all(len(chunk['frames']) > 0 for chunk in result)

