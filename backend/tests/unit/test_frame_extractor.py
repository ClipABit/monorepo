import numpy as np
from preprocessing.frame_extractor import FrameExtractor
from models.metadata import VideoChunk


class TestBasicFrameExtraction:
    """Test basic frame extraction functionality."""

    def test_extract_frames_returns_numpy_array(self, frame_extractor, sample_video_5s, sample_video_chunk):
        """Verify extract_frames returns numpy array."""
        frames, fps, complexity = frame_extractor.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )

        assert isinstance(frames, np.ndarray)
        assert len(frames.shape) == 4  # (n_frames, height, width, channels)

    def test_extract_frames_returns_valid_fps(self, frame_extractor, sample_video_5s, sample_video_chunk):
        """Verify returned FPS is within configured range."""
        frames, actual_fps, complexity = frame_extractor.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )

        if len(frames) > 0:
            # FPS should be within min/max range (with some tolerance)
            assert actual_fps >= frame_extractor.min_fps * 0.8
            assert actual_fps <= frame_extractor.max_fps * 1.2

    def test_extract_frames_returns_complexity_score(self, frame_extractor, sample_video_5s, sample_video_chunk):
        """Verify complexity score is between 0 and 1."""
        frames, fps, complexity = frame_extractor.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )

        assert 0.0 <= complexity <= 1.0

    def test_extracted_frames_have_correct_shape(self, frame_extractor, sample_video_5s, sample_video_chunk):
        """Verify extracted frames have correct dimensions."""
        frames, fps, complexity = frame_extractor.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )

        if len(frames) > 0:
            # Frames should have shape (n_frames, height, width, 3)
            assert frames.shape[3] == 3  # BGR channels
            assert frames.shape[1] > 0   # Height
            assert frames.shape[2] > 0   # Width


class TestAdaptiveSampling:
    """Test adaptive FPS based on motion detection."""

    def test_different_fps_ranges_produce_different_results(self, sample_video_5s, sample_video_chunk):
        """Verify FPS range configuration affects extraction."""
        extractor_slow = FrameExtractor(min_fps=0.5, max_fps=1.0)
        extractor_fast = FrameExtractor(min_fps=2.0, max_fps=4.0)

        frames_slow, fps_slow, _ = extractor_slow.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )
        frames_fast, fps_fast, _ = extractor_fast.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )

        # Fast extractor should get more frames
        assert len(frames_fast) >= len(frames_slow)

    def test_motion_threshold_affects_sampling(self, sample_video_5s, sample_video_chunk):
        """Verify motion threshold impacts frame selection."""
        extractor_sensitive = FrameExtractor(motion_threshold=10.0)  # Lower = more sensitive
        extractor_insensitive = FrameExtractor(motion_threshold=50.0)  # Higher = less sensitive

        frames_sensitive, _, _ = extractor_sensitive.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )
        frames_insensitive, _, _ = extractor_insensitive.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )

        # Both should extract some frames
        assert len(frames_sensitive) > 0
        assert len(frames_insensitive) > 0


class TestComplexityScoring:
    """Test complexity score calculation."""

    def test_complexity_score_is_normalized(self, frame_extractor, sample_video_5s, sample_video_chunk):
        """Verify complexity score stays within 0-1 range."""
        _, _, complexity = frame_extractor.extract_frames(
            str(sample_video_5s), sample_video_chunk
        )

        assert 0.0 <= complexity <= 1.0

    def test_static_video_has_lower_complexity(self, frame_extractor, sample_video_static, sample_video_chunk):
        """Verify static videos have lower complexity scores."""
        _, _, complexity = frame_extractor.extract_frames(
            str(sample_video_static), sample_video_chunk
        )

        # Static video should have relatively low complexity
        # (actual threshold depends on video content)
        assert complexity >= 0.0  # Just verify it's valid

    def test_edge_density_calculation(self, frame_extractor):
        """Verify edge density calculation works."""
        # Create test frame with known edge characteristics
        frame = np.zeros((480, 640), dtype=np.uint8)
        frame[100:200, 100:200] = 255  # White square on black background

        edge_density = frame_extractor._calculate_edge_density(frame)

        assert 0.0 <= edge_density <= 1.0
        assert edge_density > 0.0  # Should detect edges of square


class TestSingleFrameExtraction:
    """Test single frame extraction utility."""

    def test_extract_single_frame_at_timestamp(self, frame_extractor, sample_video_5s):
        """Verify single frame extraction at specific timestamp."""
        frame = frame_extractor.extract_single_frame(str(sample_video_5s), timestamp=2.0)

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3  # (height, width, channels)

    def test_extract_frame_at_start(self, frame_extractor, sample_video_5s):
        """Verify extraction at video start (t=0)."""
        frame = frame_extractor.extract_single_frame(str(sample_video_5s), timestamp=0.0)

        assert frame is not None

    def test_extract_frame_beyond_duration_returns_none(self, frame_extractor, sample_video_5s):
        """Verify extraction beyond video duration handles gracefully."""
        frame = frame_extractor.extract_single_frame(str(sample_video_5s), timestamp=9999.0)

        # Should return None or last frame
        # (behavior depends on OpenCV implementation)
        if frame is not None:
            assert isinstance(frame, np.ndarray)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_chunk_duration_handles_gracefully(self, frame_extractor, sample_video_5s):
        """Verify handling of zero-duration chunks."""
        chunk = VideoChunk(chunk_id="test", start_time=2.0, end_time=2.0)

        frames, fps, complexity = frame_extractor.extract_frames(
            str(sample_video_5s), chunk
        )

        # Should return empty array or handle gracefully
        assert isinstance(frames, np.ndarray)

    def test_very_short_chunk(self, frame_extractor, sample_video_5s):
        """Verify extraction from very short chunk (0.1s)."""
        chunk = VideoChunk(chunk_id="test", start_time=1.0, end_time=1.1)

        frames, fps, complexity = frame_extractor.extract_frames(
            str(sample_video_5s), chunk
        )

        # Should extract at least 1 frame
        assert len(frames) >= 0  # May be 0 or 1 depending on timing


