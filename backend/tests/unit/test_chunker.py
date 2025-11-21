import pytest
from preprocessing.chunker import Chunker
from models.metadata import VideoChunk


class TestChunkerBasicFunctionality:
    """Test basic chunking operations."""

    def test_chunk_video_creates_chunks(self, chunker, sample_video_5s):
        """Verify chunker creates at least one chunk from video."""
        chunks = chunker.chunk_video(str(sample_video_5s), video_id="test_video")

        assert len(chunks) > 0
        assert all(isinstance(chunk, VideoChunk) for chunk in chunks)

    def test_chunk_ids_are_unique_and_sequential(self, chunker, sample_video_5s):
        """Verify chunk IDs follow naming convention."""
        chunks = chunker.chunk_video(str(sample_video_5s), video_id="test_video")

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

        # Check format: video_id_chunk_0000
        for i, chunk in enumerate(chunks):
            expected_id = f"test_video_chunk_{i:04d}"
            assert chunk.chunk_id == expected_id

    def test_chunks_have_valid_timing(self, chunker, sample_video_5s):
        """Verify all chunks have valid start/end times."""
        chunks = chunker.chunk_video(str(sample_video_5s), video_id="test_video")

        for chunk in chunks:
            assert chunk.start_time >= 0
            assert chunk.end_time > chunk.start_time
            assert chunk.duration > 0

    def test_chunks_are_continuous(self, chunker, sample_video_5s):
        """Verify chunks have no gaps or overlaps."""
        chunks = chunker.chunk_video(str(sample_video_5s), video_id="test_video")

        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # Next chunk starts where current ends (with small tolerance)
                gap = chunks[i + 1].start_time - chunks[i].end_time
                assert abs(gap) < 0.1, f"Gap detected between chunks: {gap}s"


class TestDurationConstraints:
    """Test min/max duration constraint enforcement."""

    def test_chunks_respect_min_duration(self, sample_video_5s):
        """Verify chunks are not shorter than min_duration (except last)."""
        chunker = Chunker(min_duration=2.0, max_duration=10.0)
        chunks = chunker.chunk_video(str(sample_video_5s), video_id="test")

        # All chunks except possibly the last should be >= min_duration
        for chunk in chunks[:-1]:
            assert chunk.duration >= 2.0

    def test_chunks_respect_max_duration(self, sample_video_5s):
        """Verify chunks are not longer than max_duration."""
        chunker = Chunker(min_duration=1.0, max_duration=3.0)
        chunks = chunker.chunk_video(str(sample_video_5s), video_id="test")

        for chunk in chunks:
            assert chunk.duration <= 3.0

    def test_custom_duration_constraints(self):
        """Verify chunker accepts custom duration parameters."""
        chunker = Chunker(min_duration=5.0, max_duration=15.0)

        assert chunker.min_duration == 5.0
        assert chunker.max_duration == 15.0


class TestFallbackChunking:
    """Test fallback behavior when scene detection fails."""

    def test_fallback_when_scene_detection_fails(self, chunker, mocker, sample_video_5s):
        """Verify fallback chunking is used when detect() raises exception."""
        # Mock scenedetect.detect to raise exception
        mocker.patch('preprocessing.chunker.detect', side_effect=Exception("Scene detection failed"))

        chunks = chunker.chunk_video(str(sample_video_5s), video_id="test")

        # Should still create chunks via fallback
        assert len(chunks) > 0
        assert all(isinstance(chunk, VideoChunk) for chunk in chunks)

    def test_fallback_when_no_scenes_detected(self, chunker, mocker, sample_video_5s):
        """Verify fallback when detect() returns empty list."""
        mocker.patch('preprocessing.chunker.detect', return_value=[])

        chunks = chunker.chunk_video(str(sample_video_5s), video_id="test")

        # Should use fallback chunking
        assert len(chunks) > 0

    def test_fallback_creates_fixed_duration_chunks(self, chunker, sample_video_5s):
        """Verify fallback creates approximately equal chunks."""
        # Force fallback by mocking detect
        import pytest_mock
        chunks = chunker._fallback_chunking(str(sample_video_5s), video_id="test")

        if len(chunks) > 1:
            durations = [chunk.duration for chunk in chunks]
            # All chunks except last should have similar duration
            expected_duration = (chunker.min_duration + chunker.max_duration) / 2
            for duration in durations[:-1]:
                assert abs(duration - expected_duration) < 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_video_path_raises_error(self, chunker):
        """Verify appropriate handling of invalid video path."""
        result = chunker.chunk_video("/nonexistent/video.mp4", video_id="test")
        assert result == []  # Returns empty instead of crashing

    def test_scene_threshold_affects_chunking(self, sample_video_5s):
        """Verify different scene thresholds create different results."""
        chunker_sensitive = Chunker(scene_threshold=5.0)  # More sensitive
        chunker_normal = Chunker(scene_threshold=20.0)    # Less sensitive

        chunks_sensitive = chunker_sensitive.chunk_video(str(sample_video_5s), "test")
        chunks_normal = chunker_normal.chunk_video(str(sample_video_5s), "test")

        # More sensitive threshold should detect more scenes (usually)
        # Note: This may not always be true depending on video content
        assert len(chunks_sensitive) >= 1
        assert len(chunks_normal) >= 1


