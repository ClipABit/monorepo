"""
Search module for semantic text/video search using CLIP embeddings.

This module provides classes for:
- Text embedding generation (CLIP)
- Pinecone vector search
- Result ranking and filtering
"""

from .embedder import TextEmbedder
from .searcher import Searcher

__all__ = ['TextEmbedder', 'Searcher']
