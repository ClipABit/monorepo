"""
Dev Combined Modal App

ONLY FOR LOCAL DEVELOPMENT (ENVIRONMENT=dev)

Combines Server, Search, and Processing into one app for easy local iteration.
All three services run in the same Modal app, so no cross-app lookups needed.

For staging/prod deployments, use the separate apps:
- apps/server.py
- apps/search_app.py
- apps/processing_app.py

Note: Cold starts will be slower (~20s) since all dependencies load together,
but this is acceptable for local development where iteration speed matters more.
"""

import logging
import modal

from shared.config import get_environment, get_secrets, is_internal_env
from shared.images import get_dev_image
from services.search import SearchService
from services.processing import ProcessingService
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

DevSearchService = app.cls(cpu=2.0, memory=2048, timeout=60, scaledown_window=120)(SearchService)
DevProcessingService = app.cls(cpu=4.0, memory=4096, timeout=600)(ProcessingService)

# Define DevServer to add the asgi_app method and pass service classes
@app.cls(cpu=2.0, memory=2048, timeout=120)
class DevServer(ServerService):
    """Server with ASGI app for dev combined mode."""
    
    @modal.enter()
    def startup(self):
        """Initialize connectors and create FastAPI app with service classes."""
        self._initialize_connectors()
        # Create FastAPI app with registered service classes (dev combined mode)
        self.fastapi_app = self.create_fastapi_app(
            search_service_cls=DevSearchService,
            processing_service_cls=DevProcessingService
        )

    @modal.asgi_app()
    def asgi_app(self):
        return self.fastapi_app
