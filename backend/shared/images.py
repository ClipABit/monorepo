"""
Modal Image definitions for each app.

Separates dependencies to minimize cold start times:
- Server: Minimal deps (~3-5s cold start)
- Search: Medium deps with CLIP text encoder (~8-10s cold start)
- Processing: Heavy deps with full video pipeline (~15-20s cold start)
"""

import modal

def _download_clip_full_model_for_dev():
    """Pre-download full CLIP model for video processing at image build time."""
    from transformers import CLIPModel, CLIPProcessor
    model_name = "openai/clip-vit-base-patch32"
    # Full model for video processing
    CLIPModel.from_pretrained(model_name)
    CLIPProcessor.from_pretrained(model_name, use_fast=True)


def get_dev_image() -> modal.Image:
    """
    Create the Modal image for the dev app.

    Pre-downloads all models at build time to eliminate cold start downloads.
    Uses ONNX for text embedding (search) and PyTorch for video processing.
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
            "onnxruntime",
            "onnxscript",
            "tokenizers",
            "firebase-admin",
            "pyjwt[crypto]",
            "requests"
        )
        .run_function(_download_clip_full_model_for_dev)
        .run_function(_export_clip_text_to_onnx)
        .add_local_python_source(
            "api",
            "auth",
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
            "firebase-admin",
            "pyjwt[crypto]",
            "requests"
        )
        .add_local_python_source(
            "database",
            "models",
            "api",
            "auth",
            "shared",
            "services",
        )
    )

def _export_clip_text_to_onnx():
    """
    Export CLIP text encoder to ONNX format at image build time.

    This runs ONCE during image build (not at cold start).
    Uses PyTorch to load and export the model, then saves both:
    - The ONNX model file (~150MB)
    - The tokenizer

    At runtime, only onnxruntime is imported (no torch), so cold starts are fast.
    """
    import os
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    model_name = "openai/clip-vit-base-patch32"
    model_dir = "/root/models"
    onnx_path = f"{model_dir}/clip_text_encoder.onnx"
    tokenizer_path = f"{model_dir}/clip_tokenizer"

    os.makedirs(model_dir, exist_ok=True)

    print(f"[BUILD TIME] Loading PyTorch CLIP text model: {model_name}")
    model = CLIPTextModelWithProjection.from_pretrained(model_name)
    model.eval()

    # Create dummy inputs for ONNX export
    dummy_input_ids = torch.randint(0, 49408, (1, 77))  # vocab_size=49408, seq_len=77
    dummy_attention_mask = torch.ones(1, 77, dtype=torch.long)

    print(f"[BUILD TIME] Exporting to ONNX format: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "text_embeds": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"[BUILD TIME] ONNX model saved: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    # Save the fast tokenizer
    print(f"[BUILD TIME] Saving tokenizer to: {tokenizer_path}")
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    tokenizer.save_pretrained(tokenizer_path)

    # Verify the ONNX model works
    print("[BUILD TIME] Verifying ONNX model...")
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    test_input_ids = np.random.randint(0, 49408, (1, 77), dtype=np.int64)
    test_attention_mask = np.ones((1, 77), dtype=np.int64)
    outputs = session.run(None, {
        "input_ids": test_input_ids,
        "attention_mask": test_attention_mask
    })

    print(f"[BUILD TIME] ✓ ONNX model verified! Output shape: {outputs[0].shape}")
    print("[BUILD TIME] ✓ Export complete!")


def get_search_image() -> modal.Image:
    """
    Create the Modal image for the Search app.

    Uses ONNX Runtime and raw tokenizers for minimal cold starts.
    - No PyTorch at runtime (~2GB saved)
    - No transformers at runtime (~5-8s import time saved)
    - Only onnxruntime + tokenizers (~100MB total)

    Build strategy:
    1. Install torch + transformers temporarily (build time only)
    2. Export CLIP model to ONNX format and save tokenizer
    3. Uninstall torch + transformers to reduce image size
    4. Install lightweight runtime deps (onnxruntime, tokenizers)
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        # Step 1: Install torch for model export (build time)
        .pip_install(
            "torch",
            "transformers",
            "onnxruntime",
            "onnxscript",
        )
        # Step 2: Export model to ONNX (build time)
        .run_function(_export_clip_text_to_onnx)
        # Step 3: Remove torch and transformers to save space and import time
        .run_commands("pip uninstall -y torch transformers")
        # Step 4: Install lightweight runtime dependencies
        .pip_install(
            "pinecone",
            "boto3",
            "numpy",
            "tokenizers",
            "fastapi[standard]",
            "pyjwt[crypto]",
            "requests",
            "firebase-admin",
        )
        .add_local_python_source(
            "api",
            "auth",
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
            "fastapi[standard]",
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
