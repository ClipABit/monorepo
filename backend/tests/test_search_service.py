import sys
import types


class _Match:
    def __init__(self, _id, score, metadata):
        self.id = _id
        self.score = score
        self.metadata = metadata


class _Resp:
    def __init__(self, matches):
        self.matches = matches


def _install_pinecone_stubs():
    pinecone_mod = types.ModuleType("pinecone")

    class ServerlessSpec:
        def __init__(self, cloud: str, region: str):
            self.cloud = cloud
            self.region = region

    class _Index:
        def query(self, **kwargs):
            return _Resp([
                _Match("chunk:vid:1:visual", 0.9, {"video_id": "vid", "chunk_id": "1", "modality": "visual"}),
                _Match("chunk:vid:2:audio", 0.8, {"video_id": "vid", "chunk_id": "2", "modality": "audio"}),
            ])

    class Pinecone:
        def __init__(self, api_key=None):
            pass
        def list_indexes(self):
            return [{"name": "clip-chunks-v1"}]
        def create_index(self, *args, **kwargs):
            return None
        def Index(self, name):
            return _Index()

    pinecone_mod.ServerlessSpec = ServerlessSpec
    pinecone_mod.Pinecone = Pinecone
    sys.modules["pinecone"] = pinecone_mod

    # dotenv + deepface stubs
    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_mod
    deepface_mod = types.ModuleType("deepface")
    class DeepFace:
        pass
    deepface_mod.DeepFace = DeepFace
    sys.modules["deepface"] = deepface_mod


def test_embed_and_search_maps_results_without_running_torch_models(monkeypatch):
    _install_pinecone_stubs()

    # Defer imports until after stubbing
    import backend.search_service as ss
    import backend.embedding_service as es
    import numpy as np

    # Monkeypatch embed_text to avoid torch
    monkeypatch.setattr(es, "embed_text", lambda q: np.array([1.0, 0.0]))

    results = ss.embed_and_search("test query", top_k=3)
    assert len(results) >= 1
    assert "id" in results[0]
    assert "score" in results[0]
    assert "video_id" in results[0]


