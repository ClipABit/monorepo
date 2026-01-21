"""
Search Modal App

Handles semantic search with CLIP text encoder.
Medium-weight dependencies (~8-10s cold start) - lighter than full video processing.

Uses CLIPTextModelWithProjection (~150MB) instead of full CLIPModel (~350MB).
"""

import logging
import modal

from shared.config import get_environment, get_secrets
from shared.images import get_search_image
from services.search import SearchService

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
app.cls(cpu=2.0, memory=2048, timeout=60, scaledown_window=120)(SearchService)
