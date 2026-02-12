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
    IS_FILE_CHANGE_ENABLED = os.environ.get("IS_FILE_CHANGE_ENABLED", "true").lower() == "true"

    # Validate environment
    if ENVIRONMENT not in ["dev", "prod", "staging"]:
        raise ValueError(f"Invalid ENVIRONMENT value: {ENVIRONMENT}. Must be one of: dev, prod, staging")

    # Dev name prefix (required in dev mode to avoid naming conflicts)
    DEV_NAME = os.environ.get("DEV_NAME", "")
    if ENVIRONMENT == "dev" and not DEV_NAME:
        raise ValueError(
            "DEV_NAME environment variable is required for dev mode. "
            "Run with: uv run dev <name>"
        )

    print(f"Running in {ENVIRONMENT} environment" + (f" (dev instance: {DEV_NAME})" if DEV_NAME else ""))

    # Modal app name (matches backend app name)
    APP_NAME = f"clipabit-{ENVIRONMENT}"

    # Determine url portion based on environment
    url_portion = "dev" if ENVIRONMENT == "dev" else ""
    url_portion2 = "-dev" if ENVIRONMENT == "dev" else ""

    # App name prefix for dev mode (e.g., "john-dev" instead of "dev")
    app_prefix = f"{DEV_NAME}-{ENVIRONMENT}" if ENVIRONMENT == "dev" else ENVIRONMENT   

    # Server API URL (handles upload, status, videos, delete, cache)
    SERVER_BASE_URL = f"https://clipabit01--{app_prefix}-server-{url_portion}server-asgi-app{url_portion2}.modal.run"

    # Search API URL (in dev its server-searchservice-asgi-app, else its search-searchservice-asgi-app)
    SEARCH_BASE_URL = f"https://clipabit01--{app_prefix}-{"server" if ENVIRONMENT == "dev" else "search"}-searchservice-asgi-app{url_portion2}.modal.run"

    # API Endpoints
    SERVER_UPLOAD_URL = f"{SERVER_BASE_URL}/upload"
    SERVER_STATUS_URL = f"{SERVER_BASE_URL}/status"
    SEARCH_STATUS_URL = f"{SEARCH_BASE_URL}/status"
    SEARCH_SEARCH_URL = f"{SEARCH_BASE_URL}/search"
    SERVER_LIST_VIDEOS_URL = f"{SERVER_BASE_URL}/videos"
    SERVER_DELETE_VIDEO_URL = f"{SERVER_BASE_URL}/videos/{{hashed_identifier}}"

    # Namespace for Pinecone and R2 (web-demo for public demo)
    NAMESPACE = "web-demo"

    @classmethod
    def get_config(cls):
        """Get configuration as a dictionary."""
        return {
            # General settings
            "environment": cls.ENVIRONMENT,
            "app_name": cls.APP_NAME,
            "namespace": cls.NAMESPACE,

            # Flags
            "is_file_change_enable": cls.IS_FILE_CHANGE_ENABLED,

            # API Endpoints
            "server_base_url": cls.SERVER_BASE_URL,
            "search_base_url": cls.SEARCH_BASE_URL,
            "server_upload_url": cls.SERVER_UPLOAD_URL,
            "server_status_url": cls.SERVER_STATUS_URL,
            "search_status_url": cls.SEARCH_STATUS_URL,
            "search_search_url": cls.SEARCH_SEARCH_URL,
            "server_list_videos_url": cls.SERVER_LIST_VIDEOS_URL,
            "server_delete_video_url": cls.SERVER_DELETE_VIDEO_URL,
        }

    @classmethod
    def print_config_partial(cls):
        """Print current configuration for debugging."""
        config = cls.get_config()
        logger.info("Current Configuration:")
        logger.info(f"  Environment: {config['environment']}")
        logger.info(f"  File Change Enabled: {config['is_file_change_enable']}")
        logger.info(f"  App Name: {config['app_name']}")
        logger.info(f"  Namespace: {config['namespace']}")

    @classmethod
    def print_config_full(cls):
        """Print current configuration for debugging."""
        config = cls.get_config()
        logger.info("Current Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")