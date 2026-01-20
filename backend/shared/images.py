"""
Modal Image definitions for each app.

Separates dependencies to minimize cold start times:
- Server: Minimal deps (~3-5s cold start)
- Search: Medium deps with CLIP text encoder (~8-10s cold start)
- Processing: Heavy deps with full video pipeline (~15-20s cold start)
"""

import modal

def get_dev_image() -> modal.Image:
    """
    Create the Modal image for the dev app.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libsm6", "libxext6")
        .pip_install(
            "fastapi[standard]",
            "python-multipart",
            "boto3",
            "pinecone",
            "numpy",
            "torch",
            "torchvision",
            "transformers",
            "opencv-python-headless",
            "scenedetect",
            "pillow",
        )
        .add_local_python_source(
            "database",
            "models",
            "api",
            "shared",
            "preprocessing",
            "embeddings",
            "models",
            "search",
        )
    )

def get_server_image() -> modal.Image:
    """
    Create the Modal image for the Server app.
    
    Minimal dependencies for fast cold starts.
    Handles: health, status, upload, search, list_videos, delete operations.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "fastapi[standard]",
            "python-multipart",
            "boto3",
            "pinecone",
            "numpy",
        )
        .add_local_python_source(
            "database",
            "models",
            "api",
            "shared",
        )
    )

def get_search_image() -> modal.Image:
    """
    Create the Modal image for the Search app.
    
    Medium dependencies - includes CLIP text encoder only.
    The text encoder (~150MB) is much lighter than the full CLIP model (~350MB).
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch",
            "transformers",
            "pinecone",
            "boto3",
            "numpy",
        )
        .add_local_python_source(
            "database",
            "search",
            "shared",
        )
    )


def get_processing_image() -> modal.Image:
    """
    Create the Modal image for the Processing app.
    
    Heavy dependencies for video processing pipeline.
    Includes: ffmpeg, opencv, scenedetect, full CLIP model, etc.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libsm6", "libxext6")
        .pip_install(
            "torch",
            "torchvision",
            "transformers",
            "opencv-python-headless",
            "scenedetect",
            "pillow",
            "numpy",
            "pinecone",
            "boto3",
        )
        .add_local_python_source(
            "database",
            "preprocessing",
            "embeddings",
            "models",
            "shared",
        )
    )
