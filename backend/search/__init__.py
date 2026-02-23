"""
Search module for semantic search using CLIP embeddings and Pinecone.
"""

from search.text_embedder import TextEmbedder
from search.searcher import Searcher

__all__ = ["TextEmbedder", "Searcher"]
