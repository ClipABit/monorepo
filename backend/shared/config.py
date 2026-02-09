"""
Shared configuration module for Modal apps.

Provides environment handling, app naming, and secrets management
shared across all Modal apps (Server, Search, Processing).
"""

import os
import sys
import logging
import modal


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging to send INFO and DEBUG to stdout, WARNING+ to stderr.
    
    This ensures info logs appear in stdout for proper log routing in production.
    
    Args:
        level: Minimum logging level (default: logging.INFO)
    """
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler for INFO and DEBUG -> stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)
    stdout_handler.setFormatter(formatter)
    
    # Handler for WARNING and above -> stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)


# Configure logging on module import
configure_logging()
logger = logging.getLogger(__name__)

# Valid environments
VALID_ENVIRONMENTS = ["dev", "prod", "staging"]


def get_environment() -> str:
    """
    Get the current environment from ENVIRONMENT variable.
    
    Returns:
        str: Environment name (dev, prod, or staging)
    
    Raises:
        ValueError: If ENVIRONMENT is not a valid value
    """
    env = os.environ.get("ENVIRONMENT", "dev")
    if env not in VALID_ENVIRONMENTS:
        raise ValueError(f"Invalid ENVIRONMENT: {env}. Must be one of: {VALID_ENVIRONMENTS}")
    return env

def get_modal_environment() -> str:
    """Get the modal environment name."""
    return 'main'

def get_secrets() -> modal.Secret:
    """
    Get Modal secrets for the current environment.
    
    Returns:
        modal.Secret: Secret object containing environment variables
    """
    env = get_environment()
    return modal.Secret.from_name(env)


def get_pinecone_index() -> str:
    """
    Get the Pinecone index name for the current environment.
    
    Returns:
        str: Pinecone index name (e.g., "dev-chunks")
    """
    env = get_environment()
    return f"{env}-chunks"


# Environment variable helpers for connectors
def get_env_var(key: str) -> str:
    """
    Get a required environment variable or raise an error.
    
    Args:
        key: Environment variable name
    
    Returns:
        str: Environment variable value
    
    Raises:
        ValueError: If the environment variable is not set
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(f"{key} not found in environment variables")
    return value


