"""
Dev Combined Modal App

ONLY FOR LOCAL DEVELOPMENT (ENVIRONMENT=dev)

Combines Server, Search, and Processing into one app for easy local iteration.
All three services run in the same Modal app, so no cross-app lookups needed.

For staging/prod deployments, use the separate apps:
- apps/server.py
- apps/search_app.py
- apps/processing_app.py

Endpoints:
- DevServer: /health, /status, /upload, /videos, /videos/{id}, /cache/clear
- DevSearchService: /health, /search (separate ASGI app for lower latency)
"""

import logging
import os
import modal

from shared.config import get_environment, get_secrets
from shared.images import get_dev_image
from services.search_service import SearchService
from services.processing_service import ProcessingService
from services.http_server import ServerService

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()


if env != "dev":
    raise ValueError(
        f"dev_combined.py is ONLY for local development (ENVIRONMENT=dev). "
        f"Current environment: {env}. Use separate apps for staging/prod."
    )

# Get dev name prefix (required in dev mode to avoid naming conflicts)
# Only validate locally - Modal containers re-import but don't need DEV_NAME
# since the app name is already determined by the local import
dev_name = os.environ.get("DEV_NAME")
if not dev_name:
    if modal.is_local():
        raise ValueError(
            "DEV_NAME environment variable is required for dev mode. "
            "Run with: uv run dev <name>"
        )
    else:
        # Inside Modal container - use placeholder (app name already set)
        dev_name = "container"

app_name = f"{dev_name}-{env}-server"
logger.info(f"Starting Combined Dev App '{app_name}' - all services in one app for local iteration")

app = modal.App(
    name=app_name,
    image=get_dev_image(),
    secrets=[get_secrets()]
)

# SearchService exposes its own ASGI app for direct HTTP access (no server hop)
DevSearchService = app.cls(
    cpu=2.0,
    memory=2048,
    timeout=60,
    scaledown_window=120,
    enable_memory_snapshot=True,  # Snapshot after @enter() for faster subsequent cold starts
)(SearchService)

DevProcessingService = app.cls(cpu=4.0, memory=4096, timeout=600)(ProcessingService)


# Define DevServer to add the asgi_app method and pass service classes
@app.cls(cpu=2.0, memory=2048, timeout=120, scaledown_window=120)
class DevServer(ServerService):
    """Server with ASGI app for dev combined mode (excludes search)."""

    @modal.enter()
    def startup(self):
        """Initialize connectors and create FastAPI app with service classes."""
        self._initialize_connectors()
        # Create FastAPI app (search is handled by DevSearchService's own ASGI app)
        self.fastapi_app = self.create_fastapi_app(
            processing_service_cls=DevProcessingService
        )

    @modal.asgi_app()
    def asgi_app(self):
        return self.fastapi_app
