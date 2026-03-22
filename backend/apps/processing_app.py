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

# Shared volume for models (e.g., MediaPipe face_landmarker.task)
models_vol = modal.Volume.from_name("models")

# Create Modal app with processing-specific image
app = modal.App(
    name=f"{env}-processing",
    image=get_processing_image(),
    secrets=[get_secrets()]
)

# Register ProcessingService with this app, mounting the models volume at /models
app.cls(cpu=4.0, memory=4096, timeout=600, volumes={"/models": models_vol})(ProcessingService)
