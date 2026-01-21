"""
Modal Image definitions for each app.

Separates dependencies to minimize cold start times:
- Server: Minimal deps (~3-5s cold start)
- Search: Medium deps with CLIP text encoder (~8-10s cold start)
- Processing: Heavy deps with full video pipeline (~15-20s cold start)
"""

import modal

def _download_all_clip_models():
    """Pre-download all CLIP models at image build time."""
    from transformers import CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, CLIPTokenizer
    model_name = "openai/clip-vit-base-patch32"
    # Full model for video processing
    CLIPModel.from_pretrained(model_name)
    CLIPProcessor.from_pretrained(model_name, use_fast=True)
    # Text-only model for search
    CLIPTokenizer.from_pretrained(model_name)
    CLIPTextModelWithProjection.from_pretrained(model_name)


def get_dev_image() -> modal.Image:
    """
    Create the Modal image for the dev app.
    
    Pre-downloads all models at build time to eliminate cold start downloads.
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
        .run_function(_download_all_clip_models)
        .add_local_python_source(
            "api",
            "database",
            "embeddings",
            "models",
            "shared",
            "preprocessing",
            "search",
            "services"
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
            "services",
        )
    )

def _download_clip_text_model():
    """Pre-download CLIP text encoder at image build time."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    model_name = "openai/clip-vit-base-patch32"
    CLIPTokenizer.from_pretrained(model_name)
    CLIPTextModelWithProjection.from_pretrained(model_name)


def get_search_image() -> modal.Image:
    """
    Create the Modal image for the Search app.

    Medium dependencies - includes CLIP text encoder only.
    The text encoder (~150MB) is much lighter than the full CLIP model (~350MB).
    
    Pre-downloads the model at build time to eliminate cold start downloads.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "fastapi[standard]",
            "torch",
            "transformers",
            "pinecone",
            "boto3",
            "numpy",
        )
        .run_function(_download_clip_text_model)
        .add_local_python_source(
            "database",
            "search",
            "shared",
            "services",
        )
    )


def _download_clip_full_model():
    """Pre-download full CLIP model (vision + text) at image build time."""
    from transformers import CLIPModel, CLIPProcessor
    model_name = "openai/clip-vit-base-patch32"
    CLIPModel.from_pretrained(model_name)
    CLIPProcessor.from_pretrained(model_name, use_fast=True)


def get_processing_image() -> modal.Image:
    """
    Create the Modal image for the Processing app.
    
    Heavy dependencies for video processing pipeline.
    Includes: ffmpeg, opencv, scenedetect, full CLIP model, etc.
    
    Pre-downloads the model at build time to eliminate cold start downloads.
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
        .run_function(_download_clip_full_model)
        .add_local_python_source(
            "database",
            "preprocessing",
            "embeddings",
            "models",
            "shared",
            "services"
        )
    )
