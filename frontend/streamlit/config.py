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
    if ENVIRONMENT not in ["dev", "prod"]:
        raise ValueError(f"Invalid ENVIRONMENT value: {ENVIRONMENT}. Must be one of: dev, prod")
    
    print(f"Running in {ENVIRONMENT} environment")

    # Modal app name (matches backend app name)
    APP_NAME = ENVIRONMENT

    # API Endpoints - dynamically constructed based on environment
    # Pattern: https://clipabit01--{env}-server-{endpoint}-{env}.modal.run
    SEARCH_API_URL = f"https://clipabit01--{APP_NAME}-server-search-{APP_NAME}.modal.run"
    UPLOAD_API_URL = f"https://clipabit01--{APP_NAME}-server-upload-{APP_NAME}.modal.run"
    STATUS_API_URL = f"https://clipabit01--{APP_NAME}-server-status-{APP_NAME}.modal.run"
    LIST_VIDEOS_API_URL = f"https://clipabit01--{APP_NAME}-server-list-videos-{APP_NAME}.modal.run"

    # Namespace for Pinecone and R2 (web-demo for public demo)
    NAMESPACE = "web-demo"

    @classmethod
    def get_config(cls):
        """Get configuration as a dictionary."""
        return {
            "environment": cls.ENVIRONMENT,
            "app_name": cls.APP_NAME,
            "search_api_url": cls.SEARCH_API_URL,
            "upload_api_url": cls.UPLOAD_API_URL,
            "status_api_url": cls.STATUS_API_URL,
            "list_videos_api_url": cls.LIST_VIDEOS_API_URL,
            "namespace": cls.NAMESPACE,
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