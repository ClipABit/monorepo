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
from workers.search_worker import SearchWorker
from workers.processing_worker import ProcessingWorker
from workers.server_worker import Server

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

# Combined image with ALL dependencies
combined_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .pip_install(
        # Server deps
        "fastapi[standard]",
        "python-multipart",
        # ML deps (for both search and processing)
        "torch",
        "torchvision",
        "transformers",
        # Video processing deps
        "opencv-python-headless",
        "scenedetect",
        "pillow",
        # Storage deps
        "boto3",
        "pinecone",
        "numpy",
    )
    .add_local_python_source(
        "database",
        "preprocessing",
        "embeddings",
        "models",
        "search",
        "api",
        "shared",
        "workers",
    )
)

# Create single Modal app for all services
app = modal.App(
    name="dev monolith",
    image=combined_image,
    secrets=[get_secrets()]
)

# Register SearchWorker with this app
app.cls(cpu=2.0, memory=2048, timeout=60, scaledown_window=120)(SearchWorker)

# Register ProcessingWorker with this app
app.cls(cpu=4.0, memory=4096, timeout=600)(ProcessingWorker)


# Define ServerWithASGI to add the asgi_app method and pass worker classes
@app.cls(cpu=2.0, memory=2048, timeout=120)
class ServerWithASGI(Server):
    """Server with ASGI app for dev combined mode."""
    
    @modal.enter()
    def startup(self):
        # Call parent startup
        super().startup()
        # Create FastAPI app with worker classes (dev combined mode)
        self.fastapi_app = self.create_fastapi_app(
            search_worker_cls=SearchWorker,
            processing_worker_cls=ProcessingWorker
        )

    @modal.asgi_app()
    def asgi_app(self):
        return self.fastapi_app
