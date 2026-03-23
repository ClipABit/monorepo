"""
Processing Modal App

Handles video processing with full CLIP image encoder and preprocessing pipeline.
Heavy dependencies (~15-20s cold start) - acceptable for background jobs.

This app is spawned by the Server for video uploads.
"""

import logging
import modal

from shared.config import get_environment, get_secrets
from shared.images import get_processing_image
from services.processing_service import ProcessingService

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()
logger.info(f"Starting Processing App in '{env}' environment")

# Create Modal app with processing-specific image
app = modal.App(
    name=f"{env}-processing",
    image=get_processing_image(),
    secrets=[get_secrets()]
)

# Register ProcessingService with this app
# GPU dramatically speeds up CLIP embeddings (~10x faster: 20-50s → 2-5s per video)
app.cls(cpu=2.0, gpu="T4", memory=8192, timeout=3600)(ProcessingService)
