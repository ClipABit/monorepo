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
import modal

from shared.config import get_environment, get_secrets, is_internal_env
from shared.images import get_dev_image
from services.search_service import SearchService
from services.processing_service import ProcessingService
from services.http_server import ServerService

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()
IS_INTERNAL_ENV = is_internal_env()

if env != "dev":
    raise ValueError(
        f"dev_combined.py is ONLY for local development (ENVIRONMENT=dev). "
        f"Current environment: {env}. Use separate apps for staging/prod."
    )

logger.info("Starting Combined Dev App - all services in one app for local iteration")

app = modal.App(
    name=env,
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
