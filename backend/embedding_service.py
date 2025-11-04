import os
import numpy as np


_DEVICE = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def _get_clip_model():
    global _CLIP_MODEL, _CLIP_PROCESSOR, _DEVICE
    # Lazy imports to keep tests lightweight
    import torch  # type: ignore
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE)
        _CLIP_MODEL.eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _CLIP_MODEL, _CLIP_PROCESSOR


def embed_text(query: str) -> np.ndarray:
    """
    Return a 512-d L2-normalized CLIP text embedding for the query.
    """
    import torch  # type: ignore
    model, processor = _get_clip_model()
    inputs = processor(text=[query], return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    vec = text_features.cpu().squeeze(0).numpy()
    return vec


