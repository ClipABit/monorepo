"""
Server Modal App

Handles all HTTP endpoints with minimal dependencies for fast cold starts (~3-5s).
"""

import logging
import modal

from shared.config import get_environment, get_secrets
from shared.images import get_server_image
from workers.server_worker import Server

logger = logging.getLogger(__name__)

# Environment setup
env = get_environment()

# Create Modal app with minimal image
app = modal.App(
    name=f"{env}-api",
    image=get_server_image(),
    secrets=[get_secrets()]
)


# Define ServerWithASGI to add the asgi_app method and FastAPI setup
@app.cls(cpu=2.0, memory=2048, timeout=120)
class ServerWithASGI(Server):
    """Server with ASGI app for production deployment."""
    
    @modal.enter()
    def startup(self):
        # Call parent startup
        super().startup()
        # Create FastAPI app (no worker classes for production - uses from_name)
        self.fastapi_app = self.create_fastapi_app()

    @modal.asgi_app()
    def asgi_app(self):
        return self.fastapi_app
