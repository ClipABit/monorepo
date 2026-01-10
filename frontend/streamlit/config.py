"""Configuration module for ClipABit Streamlit frontend."""

import os
import logging
import sys

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for environment-based settings."""

    # Environment (defaults to "dev")
    ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")

    # Validate environment
    if ENVIRONMENT not in ["dev", "prod", "staging"]:
        raise ValueError(f"Invalid ENVIRONMENT value: {ENVIRONMENT}. Must be one of: dev, prod, staging")
    
    print(f"Running in {ENVIRONMENT} environment")

    url_suffix = "-dev" if ENVIRONMENT == "dev" else ""
    
    # This is the single entrypoint for the FastAPI app served by Modal's asgi_app
    API_BASE_URL = f"https://clipabit01--{ENVIRONMENT}-server-asgi-app{url_suffix}.modal.run"

    SEARCH_API_URL = f"{API_BASE_URL}/search"
    UPLOAD_API_URL = f"{API_BASE_URL}/upload"
    STATUS_API_URL = f"{API_BASE_URL}/status"
    LIST_VIDEOS_API_URL = f"{API_BASE_URL}/videos"
    DELETE_VIDEO_API_URL = f"{API_BASE_URL}/videos/{{hashed_identifier}}"

    # Application name
    APP_NAME = f"ClipABit Streamlit {ENVIRONMENT.capitalize()}"

    # Namespace for Pinecone and R2 (web-demo for public demo)
    NAMESPACE = "web-demo"

    # Flag to indicate if running in internal environment
    IS_INTERNAL_ENV = ENVIRONMENT in ["dev", "staging"]

    @classmethod
    def get_config(cls):
        """Get configuration as a dictionary."""
        return {
            # General settings
            "environment": cls.ENVIRONMENT,
            "app_name": cls.APP_NAME,
            "namespace": cls.NAMESPACE,

            # Flags
            "is_internal_env": cls.IS_INTERNAL_ENV,

            # API Endpoints
            "search_api_url": cls.SEARCH_API_URL,
            "upload_api_url": cls.UPLOAD_API_URL,
            "status_api_url": cls.STATUS_API_URL,
            "list_videos_api_url": cls.LIST_VIDEOS_API_URL,
            "delete_video_api_url": cls.DELETE_VIDEO_API_URL,
        }

    @classmethod
    def print_config_partial(cls):
        """Print current configuration for debugging."""
        config = cls.get_config()
        logger.info("Current Configuration:")
        logger.info(f"  Environment: {config['environment']}")
        logger.info(f"  App Name: {config['app_name']}")
        logger.info(f"  Namespace: {config['namespace']}")

    @classmethod
    def print_config_full(cls):
        """Print current configuration for debugging."""
        config = cls.get_config()
        logger.info("Current Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")