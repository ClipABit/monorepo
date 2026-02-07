"""
Server Modal App

Handles all HTTP endpoints with minimal dependencies for fast cold starts (~3-5s).
"""

import logging
import modal

from shared.config import get_environment, get_secrets
from shared.images import get_server_image
from services.http_server import ServerService

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()

# Create Modal app with minimal image
app = modal.App(
    name=f"{env}-server",
    image=get_server_image(),
    secrets=[get_secrets()]
)


@app.cls(cpu=2.0, memory=2048, timeout=120, scaledown_window=300, min_containers=1)
class Server(ServerService):
    """Server with ASGI app for production deployment."""
    
    @modal.enter()
    def startup(self):
        """Initialize connectors and create FastAPI app."""
        self._initialize_connectors()
        # Create FastAPI app (no service classes for production - uses from_name)
        self.fastapi_app = self.create_fastapi_app()

    @modal.asgi_app()
    def asgi_app(self):
        return self.fastapi_app
