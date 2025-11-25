# Make search a proper Python package

from .embedder import TextEmbedder
from .searcher import Searcher

__all__ = ["TextEmbedder", "Searcher"]
