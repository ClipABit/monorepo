"""
Unit tests for VideoEmbedder (CLIP-based).

Tests the video embedding functionality with mocked CLIP model and processor.
"""

import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch


class FakeBaseModelOutputWithPooling:
    """Fake HuggingFace output object to simulate newer transformers behavior."""

    def __init__(self, pooler_output: torch.Tensor):
        self.pooler_output = pooler_output
        self.last_hidden_state = torch.randn(1, 50, 768)


class FakeCLIPModel:
    """Fake CLIP model for testing."""

    def __init__(self, return_output_object: bool = False):
        self.return_output_object = return_output_object
        self.get_image_features_calls = []

    def to(self, device):
        return self

    def get_image_features(self, **inputs):
        self.get_image_features_calls.append(inputs)
        batch_size = inputs["pixel_values"].shape[0]
        embeddings = torch.randn(batch_size, 512)

        if self.return_output_object:
            return FakeBaseModelOutputWithPooling(embeddings)
        return embeddings


class FakeProcessorOutput:
    """Fake processor output that supports .to() method."""

    def __init__(self, pixel_values: torch.Tensor):
        self.pixel_values = pixel_values
        self._data = {"pixel_values": pixel_values}

    def to(self, device):
        return self

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)


class FakeCLIPProcessor:
    """Fake CLIP processor for testing."""

    def __init__(self):
        self.call_args = []

    def __call__(self, images, return_tensors, size):
        self.call_args.append((images, return_tensors, size))
        batch_size = len(images)
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        return FakeProcessorOutput(pixel_values)


@pytest.fixture
def mock_transformers_tensor_output():
    """Mock transformers module with tensor output (older behavior)."""
    mock_transformers = MagicMock()

    fake_model = FakeCLIPModel(return_output_object=False)
    fake_processor = FakeCLIPProcessor()

    mock_transformers.CLIPModel.from_pretrained.return_value = fake_model
    mock_transformers.CLIPProcessor.from_pretrained.return_value = fake_processor

    with patch.dict(sys.modules, {'transformers': mock_transformers}):
        yield mock_transformers, fake_model, fake_processor


@pytest.fixture
def mock_transformers_output_object():
    """Mock transformers module with BaseModelOutputWithPooling output (newer behavior)."""
    mock_transformers = MagicMock()

    fake_model = FakeCLIPModel(return_output_object=True)
    fake_processor = FakeCLIPProcessor()

    mock_transformers.CLIPModel.from_pretrained.return_value = fake_model
    mock_transformers.CLIPProcessor.from_pretrained.return_value = fake_processor

    with patch.dict(sys.modules, {'transformers': mock_transformers}):
        yield mock_transformers, fake_model, fake_processor


@pytest.fixture
def embedder_with_tensor_output(mock_transformers_tensor_output):
    """Create VideoEmbedder with mocked dependencies returning tensor."""
    mock_transformers, fake_model, fake_processor = mock_transformers_tensor_output

    from embeddings.video_embedder import VideoEmbedder
    embedder = VideoEmbedder()

    return embedder, fake_model, fake_processor


@pytest.fixture
def embedder_with_output_object(mock_transformers_output_object):
    """Create VideoEmbedder with mocked dependencies returning output object."""
    mock_transformers, fake_model, fake_processor = mock_transformers_output_object

    from embeddings.video_embedder import VideoEmbedder
    embedder = VideoEmbedder()

    return embedder, fake_model, fake_processor


class TestVideoEmbedderInitialization:
    """Test VideoEmbedder initialization."""

    def test_initializes_with_correct_device(self, mock_transformers_tensor_output):
        """Verify device is set based on CUDA availability."""
        from embeddings.video_embedder import VideoEmbedder
        embedder = VideoEmbedder()

        assert embedder._device in ["cuda", "cpu"]

    def test_loads_clip_model_on_init(self, mock_transformers_tensor_output):
        """Verify CLIP model is loaded during initialization."""
        mock_transformers, _, _ = mock_transformers_tensor_output

        from embeddings.video_embedder import VideoEmbedder
        embedder = VideoEmbedder()

        mock_transformers.CLIPModel.from_pretrained.assert_called_once()
        mock_transformers.CLIPProcessor.from_pretrained.assert_called_once()


class TestGenerateClipEmbedding:
    """Test _generate_clip_embedding functionality."""

    def test_returns_tensor(self, embedder_with_tensor_output, sample_frames):
        """Verify embedding is returned as a tensor."""
        embedder, _, _ = embedder_with_tensor_output

        result = embedder._generate_clip_embedding(sample_frames)

        assert isinstance(result, torch.Tensor)

    def test_returns_1d_embedding(self, embedder_with_tensor_output, sample_frames):
        """Verify embedding is 1D (single video embedding)."""
        embedder, _, _ = embedder_with_tensor_output

        result = embedder._generate_clip_embedding(sample_frames)

        assert result.ndim == 1
        assert result.shape == (512,)

    def test_embedding_is_normalized(self, embedder_with_tensor_output, sample_frames):
        """Verify embedding is L2 normalized."""
        embedder, _, _ = embedder_with_tensor_output

        result = embedder._generate_clip_embedding(sample_frames)
        norm = torch.linalg.norm(result)

        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_handles_output_object_from_newer_transformers(self, embedder_with_output_object, sample_frames):
        """
        Verify embedding works when model returns BaseModelOutputWithPooling.

        This test catches the regression where newer transformers versions
        return an output object instead of a raw tensor.
        """
        embedder, _, _ = embedder_with_output_object

        result = embedder._generate_clip_embedding(sample_frames)

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 1
        assert result.shape == (512,)

    def test_samples_frames_evenly(self, embedder_with_tensor_output):
        """Verify frames are sampled evenly across the video."""
        embedder, _, fake_processor = embedder_with_tensor_output

        frames = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
        embedder._generate_clip_embedding(frames, num_frames=8)

        assert len(fake_processor.call_args) == 1
        images, _, _ = fake_processor.call_args[0]
        assert len(images) == 8

    def test_handles_fewer_frames_than_requested(self, embedder_with_tensor_output):
        """Verify it handles videos with fewer frames than num_frames."""
        embedder, _, fake_processor = embedder_with_tensor_output

        frames = np.random.randint(0, 255, (3, 480, 640, 3), dtype=np.uint8)
        embedder._generate_clip_embedding(frames, num_frames=8)

        images, _, _ = fake_processor.call_args[0]
        assert len(images) == 3

    def test_calls_model_with_correct_inputs(self, embedder_with_tensor_output, sample_frames):
        """Verify model is called with processed inputs."""
        embedder, fake_model, _ = embedder_with_tensor_output

        embedder._generate_clip_embedding(sample_frames)

        assert len(fake_model.get_image_features_calls) == 1
        call_inputs = fake_model.get_image_features_calls[0]
        assert "pixel_values" in call_inputs

    def test_returns_cpu_tensor(self, embedder_with_tensor_output, sample_frames):
        """Verify result is moved to CPU."""
        embedder, _, _ = embedder_with_tensor_output

        result = embedder._generate_clip_embedding(sample_frames)

        assert result.device.type == "cpu"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_frame_video(self, embedder_with_tensor_output):
        """Verify single frame video can be embedded."""
        embedder, _, _ = embedder_with_tensor_output

        frames = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)
        result = embedder._generate_clip_embedding(frames)

        assert result.shape == (512,)
        assert torch.isclose(torch.linalg.norm(result), torch.tensor(1.0), atol=1e-5)

    def test_large_number_of_frames(self, embedder_with_tensor_output):
        """Verify large videos are handled with frame sampling."""
        embedder, _, fake_processor = embedder_with_tensor_output

        frames = np.random.randint(0, 255, (1000, 480, 640, 3), dtype=np.uint8)
        result = embedder._generate_clip_embedding(frames, num_frames=8)

        images, _, _ = fake_processor.call_args[0]
        assert len(images) == 8
        assert result.shape == (512,)


@pytest.fixture
def sample_frames():
    """Array of 10 test frames. Shape: (10, 480, 640, 3)"""
    return np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
