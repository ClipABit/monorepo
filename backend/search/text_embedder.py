"""
ONNX-based Text Embedding module using CLIP model.

Provides text-to-vector conversion using OpenAI's CLIP model
with ONNX Runtime for fast inference and minimal dependencies.

This eliminates the need for PyTorch (~2GB) and transformers library,
reducing cold start times significantly.
"""

import logging
from pathlib import Path
from typing import Union, List
import numpy as np

logger = logging.getLogger(__name__)

# Default path where ONNX model is stored in the Modal image
DEFAULT_ONNX_MODEL_PATH = "/root/models/clip_text_encoder.onnx"
DEFAULT_TOKENIZER_PATH = "/root/models/clip_tokenizer/tokenizer.json"


class TextEmbedder:
    """
    ONNX-based CLIP text embedder for semantic search.

    Converts text queries into 512-dimensional embeddings using
    OpenAI's CLIP text model (ViT-B/32 variant) via ONNX Runtime.

    Uses raw tokenizers library instead of transformers for faster imports.

    Usage:
        embedder = TextEmbedder()
        vector = embedder.embed_text("woman on a train")
    """

    def __init__(
        self,
        model_path: str = DEFAULT_ONNX_MODEL_PATH,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH
    ):
        """
        Initialize the ONNX text embedder.

        Args:
            model_path: Path to the ONNX model file.
            tokenizer_path: Path to the tokenizer.json file.
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.session = None
        self.tokenizer = None

        logger.info(f"TextEmbedder initialized (model: {model_path})")

    def _load_model(self):
        """Lazy load the ONNX model and tokenizer on first use."""
        if self.session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            logger.info(f"Loading ONNX model from: {self.model_path}")

            # Configure ONNX Runtime for CPU (no CUDA dependency)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )

            # Load tokenizer directly from tokenizers library (no transformers import)
            logger.info(f"Loading tokenizer from: {self.tokenizer_path}")
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

            # Configure padding and truncation for CLIP (max 77 tokens)
            from tokenizers import processors
            self.tokenizer.enable_padding(length=77, pad_id=0)
            self.tokenizer.enable_truncation(max_length=77)

            logger.info("ONNX model and tokenizer loaded successfully")

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text input(s).

        Args:
            text: Single text string or list of text strings

        Returns:
            numpy array of embeddings (512-d, L2-normalized)
            Shape: (512,) for single text, (N, 512) for batch
        """
        self._load_model()

        # Handle single string
        single_input = isinstance(text, str)
        if single_input:
            text = [text]

        # Tokenize inputs using raw tokenizers library
        encoded = self.tokenizer.encode_batch(text)

        # Extract input_ids and attention_mask
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        # Run ONNX inference
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        outputs = self.session.run(None, onnx_inputs)

        # Handle different ONNX model output formats
        # The model might output multiple tensors, we need the one with shape (batch_size, 512)
        text_embeds = None
        for output in outputs:
            if len(output.shape) == 2 and output.shape[1] == 512:
                text_embeds = output
                break

        if text_embeds is None:
            # Fallback: take first output if none match expected shape
            logger.warning(f"Could not find 512-d output, using first output with shape {outputs[0].shape}")
            text_embeds = outputs[0]

        # L2 normalize (essential for cosine similarity search)
        norms = np.linalg.norm(text_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / norms

        # Return single vector if single input
        if single_input:
            return text_embeds[0]

        return text_embeds
