"""
Unit tests for TextEmbedder (ONNX-based).

Tests the text embedding functionality with mocked ONNX runtime and tokenizer.
"""

import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from search.text_embedder import TextEmbedder, DEFAULT_ONNX_MODEL_PATH, DEFAULT_TOKENIZER_PATH


class FakeEncoding:
    """Fake tokenizer encoding result."""

    def __init__(self, ids: list[int], attention_mask: list[int]):
        self.ids = ids
        self.attention_mask = attention_mask


class FakeTokenizer:
    """Fake tokenizer for testing."""

    def __init__(self):
        self.padding_enabled = False
        self.truncation_enabled = False
        self.padding_length = None
        self.max_length = None

    def enable_padding(self, length: int, pad_id: int):
        self.padding_enabled = True
        self.padding_length = length

    def enable_truncation(self, max_length: int):
        self.truncation_enabled = True
        self.max_length = max_length

    def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
        """Return fake encodings for batch."""
        return [
            FakeEncoding(
                ids=[101] + [1000 + i for i in range(min(len(text.split()), 75))] + [102] + [0] * max(0, 77 - min(len(text.split()), 75) - 2),
                attention_mask=[1] * (min(len(text.split()), 75) + 2) + [0] * max(0, 77 - min(len(text.split()), 75) - 2)
            )
            for text in texts
        ]


class FakeOnnxSession:
    """Fake ONNX inference session."""

    def __init__(self):
        self.run_calls = []

    def run(self, output_names, inputs):
        self.run_calls.append((output_names, inputs))
        batch_size = inputs["input_ids"].shape[0]
        # Return random 512-d embeddings
        embeddings = np.random.randn(batch_size, 512).astype(np.float32)
        return [embeddings]


@pytest.fixture
def mock_onnx_and_tokenizer():
    """Mock onnxruntime and tokenizers modules."""
    mock_ort = MagicMock()
    mock_tokenizers = MagicMock()

    fake_session = FakeOnnxSession()
    fake_tokenizer = FakeTokenizer()

    mock_ort.SessionOptions.return_value = MagicMock()
    mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = "ORT_ENABLE_ALL"
    mock_ort.InferenceSession.return_value = fake_session

    mock_tokenizers.Tokenizer.from_file.return_value = fake_tokenizer

    with patch.dict(sys.modules, {
        'onnxruntime': mock_ort,
        'tokenizers': mock_tokenizers,
    }):
        yield mock_ort, mock_tokenizers, fake_session, fake_tokenizer


@pytest.fixture
def embedder_with_mocks(mock_onnx_and_tokenizer):
    """Create TextEmbedder with mocked dependencies."""
    mock_ort, mock_tokenizers, fake_session, fake_tokenizer = mock_onnx_and_tokenizer

    embedder = TextEmbedder(
        model_path="/test/model.onnx",
        tokenizer_path="/test/tokenizer.json"
    )
    embedder._load_model()

    return embedder, fake_session, fake_tokenizer, mock_ort, mock_tokenizers


class TestTextEmbedderInitialization:
    """Test TextEmbedder initialization."""

    def test_uses_default_paths(self):
        """Verify default paths are set correctly."""
        embedder = TextEmbedder()

        assert embedder.model_path == DEFAULT_ONNX_MODEL_PATH
        assert embedder.tokenizer_path == DEFAULT_TOKENIZER_PATH

    def test_accepts_custom_paths(self):
        """Verify custom paths are stored."""
        embedder = TextEmbedder(
            model_path="/custom/model.onnx",
            tokenizer_path="/custom/tokenizer.json"
        )

        assert embedder.model_path == "/custom/model.onnx"
        assert embedder.tokenizer_path == "/custom/tokenizer.json"

    def test_session_and_tokenizer_not_loaded_on_init(self):
        """Verify lazy loading - session/tokenizer not loaded until needed."""
        embedder = TextEmbedder()

        assert embedder.session is None
        assert embedder.tokenizer is None


class TestModelLoading:
    """Test model loading behavior."""

    def test_load_model_sets_session(self, mock_onnx_and_tokenizer):
        """Verify _load_model sets the session."""
        mock_ort, mock_tokenizers, fake_session, fake_tokenizer = mock_onnx_and_tokenizer

        embedder = TextEmbedder(
            model_path="/test/model.onnx",
            tokenizer_path="/test/tokenizer.json"
        )
        embedder._load_model()

        assert embedder.session is fake_session
        assert embedder.tokenizer is fake_tokenizer

    def test_load_model_configures_tokenizer_padding(self, mock_onnx_and_tokenizer):
        """Verify tokenizer padding is configured for CLIP (77 tokens)."""
        _, _, _, fake_tokenizer = mock_onnx_and_tokenizer

        embedder = TextEmbedder()
        embedder._load_model()

        assert fake_tokenizer.padding_enabled is True
        assert fake_tokenizer.padding_length == 77

    def test_load_model_configures_tokenizer_truncation(self, mock_onnx_and_tokenizer):
        """Verify tokenizer truncation is configured for CLIP (77 tokens)."""
        _, _, _, fake_tokenizer = mock_onnx_and_tokenizer

        embedder = TextEmbedder()
        embedder._load_model()

        assert fake_tokenizer.truncation_enabled is True
        assert fake_tokenizer.max_length == 77

    def test_load_model_only_loads_once(self, mock_onnx_and_tokenizer):
        """Verify model is only loaded once even if _load_model called multiple times."""
        mock_ort, mock_tokenizers, _, _ = mock_onnx_and_tokenizer

        embedder = TextEmbedder()
        embedder._load_model()
        embedder._load_model()
        embedder._load_model()

        # Should only be called once
        assert mock_ort.InferenceSession.call_count == 1
        assert mock_tokenizers.Tokenizer.from_file.call_count == 1


class TestEmbedText:
    """Test text embedding functionality."""

    def test_embed_single_text_returns_1d_array(self, embedder_with_mocks):
        """Verify single text returns 1D array."""
        embedder, _, _, _, _ = embedder_with_mocks

        result = embedder.embed_text("hello world")

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (512,)

    def test_embed_batch_returns_2d_array(self, embedder_with_mocks):
        """Verify batch input returns 2D array."""
        embedder, _, _, _, _ = embedder_with_mocks

        result = embedder.embed_text(["hello", "world", "test"])

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (3, 512)

    def test_embed_text_is_normalized(self, embedder_with_mocks):
        """Verify output is L2 normalized."""
        embedder, _, _, _, _ = embedder_with_mocks

        result = embedder.embed_text("hello world")
        norm = np.linalg.norm(result)

        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_embed_batch_all_normalized(self, embedder_with_mocks):
        """Verify all batch embeddings are L2 normalized."""
        embedder, _, _, _, _ = embedder_with_mocks

        result = embedder.embed_text(["hello", "world", "test"])

        for i in range(3):
            norm = np.linalg.norm(result[i])
            assert np.isclose(norm, 1.0, atol=1e-5)

    def test_embed_text_calls_tokenizer(self, embedder_with_mocks):
        """Verify tokenizer is called with input text."""
        embedder, _, fake_tokenizer, _, _ = embedder_with_mocks

        # Spy on encode_batch
        original_encode = fake_tokenizer.encode_batch
        call_args = []

        def spy_encode(texts):
            call_args.append(texts)
            return original_encode(texts)

        fake_tokenizer.encode_batch = spy_encode

        embedder.embed_text("test query")

        assert len(call_args) == 1
        assert call_args[0] == ["test query"]

    def test_embed_text_calls_session_run(self, embedder_with_mocks):
        """Verify ONNX session.run is called."""
        embedder, fake_session, _, _, _ = embedder_with_mocks

        embedder.embed_text("test")

        assert len(fake_session.run_calls) == 1
        _, inputs = fake_session.run_calls[0]
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

    def test_embed_text_input_ids_shape(self, embedder_with_mocks):
        """Verify input_ids has correct shape (batch, 77)."""
        embedder, fake_session, _, _, _ = embedder_with_mocks

        embedder.embed_text("test")

        _, inputs = fake_session.run_calls[0]
        assert inputs["input_ids"].shape[0] == 1  # batch size
        assert inputs["attention_mask"].shape[0] == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_embed_empty_string(self, embedder_with_mocks):
        """Verify empty string can be embedded."""
        embedder, _, _, _, _ = embedder_with_mocks

        result = embedder.embed_text("")

        assert result.shape == (512,)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-5)

    def test_embed_single_item_list(self, embedder_with_mocks):
        """Verify single-item list returns 2D array."""
        embedder, _, _, _, _ = embedder_with_mocks

        result = embedder.embed_text(["single"])

        assert result.ndim == 2
        assert result.shape == (1, 512)

    def test_embed_long_text_handled(self, embedder_with_mocks):
        """Verify long text is handled (truncation configured)."""
        embedder, _, _, _, _ = embedder_with_mocks

        # Very long text that would exceed 77 tokens
        long_text = " ".join(["word"] * 200)
        result = embedder.embed_text(long_text)

        assert result.shape == (512,)

    def test_embed_special_characters(self, embedder_with_mocks):
        """Verify special characters are handled."""
        embedder, _, _, _, _ = embedder_with_mocks

        result = embedder.embed_text("hello! @#$%^&*() world?")

        assert result.shape == (512,)


class TestMultipleOutputFormats:
    """Test handling of different ONNX output formats."""

    def test_handles_multiple_outputs_finds_512d(self, mock_onnx_and_tokenizer):
        """Verify correct output is selected when model returns multiple outputs."""
        mock_ort, mock_tokenizers, _, fake_tokenizer = mock_onnx_and_tokenizer

        # Create session that returns multiple outputs
        class MultiOutputSession:
            def run(self, output_names, inputs):
                return [
                    np.random.randn(1, 77, 768).astype(np.float32),  # Hidden states
                    np.random.randn(1, 512).astype(np.float32),      # Text embeds (correct)
                    np.random.randn(1, 768).astype(np.float32),      # Other output
                ]

        mock_ort.InferenceSession.return_value = MultiOutputSession()

        embedder = TextEmbedder()
        result = embedder.embed_text("test")

        # Should find the 512-d output
        assert result.shape == (512,)

    def test_fallback_to_first_output(self, mock_onnx_and_tokenizer):
        """Verify fallback to first output if no 512-d output found."""
        mock_ort, mock_tokenizers, _, fake_tokenizer = mock_onnx_and_tokenizer

        # Create session that returns unexpected shape
        class UnexpectedOutputSession:
            def run(self, output_names, inputs):
                return [
                    np.random.randn(1, 768).astype(np.float32),  # Wrong dimension
                ]

        mock_ort.InferenceSession.return_value = UnexpectedOutputSession()

        embedder = TextEmbedder()
        result = embedder.embed_text("test")

        # Falls back to first output (768-d)
        assert result.shape == (768,)
