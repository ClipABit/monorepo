import sys
import types
import math
import os


class _MockIndex:
    def __init__(self):
        self.upserts = []
        self.last_query_vector = None
        self.last_query_kwargs = None

    def upsert(self, vectors):
        # vectors: list of (id, values, metadata)
        self.upserts.extend(vectors)

    def query(self, vector, top_k=10, filter=None, include_metadata=True, include_values=False):
        self.last_query_vector = vector
        self.last_query_kwargs = {
            "top_k": top_k,
            "filter": filter,
            "include_metadata": include_metadata,
            "include_values": include_values,
        }

        class _Match:
            def __init__(self, _id, score, metadata):
                self.id = _id
                self.score = score
                self.metadata = metadata

        class _Resp:
            def __init__(self):
                self.matches = [
                    _Match("chunk:vid:1:visual", 0.9, {"video_id": "vid", "chunk_id": "1"}),
                    _Match("chunk:vid:2:audio", 0.8, {"video_id": "vid", "chunk_id": "2"}),
                ]

        return _Resp()


class _MockPineconeClient:
    def __init__(self):
        self._indexes = {}
        self._existing = []

    def list_indexes(self):
        return [{"name": n} for n in self._existing]

    def create_index(self, name, dimension, metric, spec):
        self._existing.append(name)
        self._indexes[name] = _MockIndex()

    def Index(self, name):
        if name not in self._indexes:
            self._indexes[name] = _MockIndex()
        return self._indexes[name]


def _install_stubs():
    # Stub pinecone module
    pinecone_mod = types.ModuleType("pinecone")

    class ServerlessSpec:
        def __init__(self, cloud: str, region: str):
            self.cloud = cloud
            self.region = region

    pinecone_mod.ServerlessSpec = ServerlessSpec

    class Pinecone:
        def __init__(self, api_key=None):
            self._client = _MockPineconeClient()
        def list_indexes(self):
            return self._client.list_indexes()
        def create_index(self, name, dimension, metric, spec):
            return self._client.create_index(name, dimension, metric, spec)
        def Index(self, name):
            return self._client.Index(name)

    pinecone_mod.Pinecone = Pinecone
    sys.modules["pinecone"] = pinecone_mod

    # Stub dotenv
    dotenv_mod = types.ModuleType("dotenv")
    def _load_dotenv():
        return None
    dotenv_mod.load_dotenv = _load_dotenv
    sys.modules["dotenv"] = dotenv_mod

    # DeepFace stub (unused here)
    deepface_mod = types.ModuleType("deepface")
    class DeepFace:
        pass
    deepface_mod.DeepFace = DeepFace
    sys.modules["deepface"] = deepface_mod


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def test_ensure_index_auto_creates_and_noops_on_second_call():
    _install_stubs()
    # import after stubs
    import backend.pinecone_connect as pc

    client = pc.connect_to_pinecone()
    pc.ensure_index(client, name="clip-chunks-test", dimension=512, metric="cosine")
    # second call should not raise and should not duplicate
    pc.ensure_index(client, name="clip-chunks-test", dimension=512, metric="cosine")


def test_upsert_visual_chunk_normalizes_and_uses_expected_id():
    _install_stubs()
    import numpy as np
    import backend.pinecone_connect as pc

    client = pc.connect_to_pinecone()
    idx = pc.get_index(client, name="clip-chunks-test")

    vec = np.array([3.0, 4.0])  # length 5 -> unit after normalize
    pc.upsert_visual_chunk(
        client,
        video_id="vidA",
        chunk_id="7",
        start_ts=1.23,
        end_ts=4.56,
        vector_512=vec,
        metadata={"preview_frame_uri": "s3://x"},
        index_name="clip-chunks-test",
    )

    assert len(idx.upserts) == 1
    rec_id, values, meta = idx.upserts[0]
    assert rec_id == "chunk:vidA:7:visual"
    # normalized
    assert abs(_norm(values) - 1.0) < 1e-6
    assert meta["video_id"] == "vidA"
    assert meta["modality"] == "visual"
    assert meta["start_ts"] == 1.23


def test_query_vectors_passes_normalized_vector_and_returns_matches():
    _install_stubs()
    import numpy as np
    import backend.pinecone_connect as pc

    client = pc.connect_to_pinecone()
    idx = pc.get_index(client, name="clip-chunks-test")
    q = np.array([10.0, 0.0])
    resp = pc.query_vectors(
        client, q, top_k=5, metadata_filter={"video_id": {"$eq": "vid"}}, index_name="clip-chunks-test"
    )
    # vector used in last query should be unit length
    assert abs(_norm(idx.last_query_vector) - 1.0) < 1e-6
    assert len(resp.matches) >= 1


