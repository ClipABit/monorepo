"""
Unit tests for Compressor class.

Tests frame resizing, compression, and memory optimization.
"""

import numpy as np
from preprocessing.compressor import Compressor


class TestBasicCompression:
    """Test basic compression functionality."""

    def test_compress_single_frame(self, compressor, sample_frame):
        """Verify single frame compression works."""
        frames = np.array([sample_frame])  # Add batch dimension
        compressed = compressor.compress_frames(frames)

        assert isinstance(compressed, np.ndarray)
        assert len(compressed) == 1

    def test_compress_multiple_frames(self, compressor, sample_frames):
        """Verify batch frame compression."""
        compressed = compressor.compress_frames(sample_frames)

        assert len(compressed) == len(sample_frames)
        assert compressed.shape[0] == sample_frames.shape[0]

    def test_compressed_frames_have_target_resolution(self, compressor):
        """Verify output resolution matches configuration."""
        frames = np.random.randint(0, 255, (5, 1080, 1920, 3), dtype=np.uint8)

        compressed = compressor.compress_frames(frames)

        assert compressed.shape[1] == compressor.target_height  # 480
        assert compressed.shape[2] == compressor.target_width   # 640

    def test_compressed_frames_maintain_channels(self, compressor, sample_frames):
        """Verify BGR channels are preserved."""
        compressed = compressor.compress_frames(sample_frames)

        assert compressed.shape[3] == 3  # BGR channels


class TestCompressionRatio:
    """Test compression ratio calculations."""

    def test_downscaling_reduces_size(self, compressor):
        """Verify downscaling reduces memory footprint."""
        large_frames = np.random.randint(0, 255, (10, 1080, 1920, 3), dtype=np.uint8)

        compressed = compressor.compress_frames(large_frames)

        original_size = large_frames.nbytes
        compressed_size = compressed.nbytes

        assert compressed_size < original_size

    def test_get_compression_ratio(self, compressor):
        """Verify compression ratio calculation."""
        original_shape = (10, 1080, 1920, 3)
        compressed_shape = (10, 480, 640, 3)

        ratio = compressor.get_compression_ratio(original_shape, compressed_shape)

        # Ratio should be > 1 (original is larger)
        expected_ratio = (1080 * 1920) / (480 * 640)
        assert abs(ratio - expected_ratio) < 0.1

    def test_same_resolution_gives_ratio_one(self, compressor):
        """Verify identical shapes give ratio of 1.0."""
        shape = (10, 640, 480, 3)
        ratio = compressor.get_compression_ratio(shape, shape)

        assert abs(ratio - 1.0) < 0.01


class TestResolutionHandling:
    """Test different input resolutions."""

    def test_upscaling_smaller_frames(self, compressor):
        """Verify upscaling works for smaller input frames."""
        small_frames = np.random.randint(0, 255, (5, 240, 320, 3), dtype=np.uint8)

        compressed = compressor.compress_frames(small_frames)

        # Should upscale to target resolution
        assert compressed.shape[1] == compressor.target_height
        assert compressed.shape[2] == compressor.target_width

    def test_aspect_ratio_preservation(self, compressor):
        """Verify aspect ratio handling during resize."""
        # Non-standard aspect ratio
        frames = np.random.randint(0, 255, (3, 720, 1280, 3), dtype=np.uint8)

        compressed = compressor.compress_frames(frames)

        # Should match target dimensions (may not preserve aspect ratio)
        assert compressed.shape[1] == compressor.target_height
        assert compressed.shape[2] == compressor.target_width

    def test_square_frames(self, compressor):
        """Verify square frame handling."""
        square_frames = np.random.randint(0, 255, (5, 720, 720, 3), dtype=np.uint8)

        compressed = compressor.compress_frames(square_frames)

        assert compressed.shape[1] == compressor.target_height
        assert compressed.shape[2] == compressor.target_width


class TestCustomConfiguration:
    """Test custom compressor configurations."""

    def test_custom_target_resolution(self):
        """Verify custom resolution configuration."""
        compressor = Compressor(target_width=320, target_height=240)

        frames = np.random.randint(0, 255, (3, 480, 640, 3), dtype=np.uint8)
        compressed = compressor.compress_frames(frames)

        assert compressed.shape[1] == 240
        assert compressed.shape[2] == 320

    def test_high_resolution_target(self):
        """Verify high resolution compression."""
        compressor = Compressor(target_width=1280, target_height=720)

        frames = np.random.randint(0, 255, (2, 1080, 1920, 3), dtype=np.uint8)
        compressed = compressor.compress_frames(frames)

        assert compressed.shape[1] == 720
        assert compressed.shape[2] == 1280


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_frame_array(self, compressor):
        """Verify handling of empty input."""
        empty_frames = np.array([])

        compressed = compressor.compress_frames(empty_frames)

        # Should return empty array or raise clear error
        assert isinstance(compressed, np.ndarray)

    def test_single_pixel_frame(self, compressor):
        """Verify handling of minimal frame."""
        # Need 4D shape: (n_frames, height, width, channels)
        tiny_frames = np.array([[[[255, 255, 255]]]], dtype=np.uint8)  # 1x1x1x3

        compressed = compressor.compress_frames(tiny_frames)

        assert compressed.shape[0] == 1  # 1 frame
        assert compressed.shape[1] == compressor.target_height
        assert compressed.shape[2] == compressor.target_width
        assert compressed.shape[3] == 3  # RGB channels

    def test_preserve_data_type(self, compressor, sample_frames):
        """Verify output dtype matches input."""
        compressed = compressor.compress_frames(sample_frames)

        assert compressed.dtype == sample_frames.dtype  # uint8
