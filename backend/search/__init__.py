"""
Search module for semantic search using CLIP embeddings and Pinecone.
"""

from search.embedder import TextEmbedder
from search.searcher import Searcher

__all__ = ["TextEmbedder", "Searcher"]
