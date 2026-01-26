"""
Search Modal App

Handles semantic search with CLIP text encoder.
Optimized for fast cold starts:
- ONNX Runtime instead of PyTorch (~2GB saved)
- Raw tokenizers instead of transformers (~5-8s import time saved)
- Memory snapshots for instant subsequent cold starts

Uses exported CLIP text model in ONNX format.
"""

import logging
import modal

from shared.config import get_environment, get_secrets
from shared.images import get_search_image
from services.search_service import SearchService

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()
logger.info(f"Starting Search App in '{env}' environment")

# Create Modal app with search-specific image
app = modal.App(
    name=f"{env}-search",
    image=get_search_image(),
    secrets=[get_secrets()]
)

# Register SearchService with this app
app.cls(
    cpu=2.0,
    memory=2048,
    timeout=60,
    scaledown_window=120,
    enable_memory_snapshot=True,  # Snapshot after @enter() for faster subsequent cold starts
)(SearchService)
