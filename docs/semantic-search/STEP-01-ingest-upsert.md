## Step 01 — Ingest and Upsert Chunk/Face Embeddings into Pinecone

Goal: ensure there is data in Pinecone to compare against a query. This step covers index setup and batch upsert for visual, audio (transcript), and face cluster vectors.

### Prereqs
- Env: `PINECONE_API_KEY`, `PINECONE_INDEX_CHUNKS=clip-chunks-v1`, `PINECONE_INDEX_FACES=faces-v1`
- Install: `pip install pinecone-client torch transformers numpy python-dotenv`
- CLIP embeddings from your pipeline or from `embeddings.py` (e.g., `generate_clip_embedding`, `generate_clip_text_embedding`).

### Single-chunk mode (your scale)
- We will handle one chunk at a time end-to-end. Batch size can be `1` safely.
- No per-frame vectors are stored; only the averaged per-chunk vector plus optional face cluster representatives.
- Keep `top_k` small during testing (≤ 25) to minimize read units.

### Index design
- `clip-chunks-v1`: dim=512, metric=cosine (CLIP text/image)
- Faces are written into the same index with `modality="face"` using 512-d face embeddings (ArcFace/Facenet512 recommended) to keep one index only.

### ID and metadata schema
- id formats:
  - chunk: `chunk:{video_id}:{chunk_id}:{modality}` where modality ∈ {visual,audio,combined}
  - face: `face:{video_id}:{cluster_id}`
- common metadata: `video_id`, `chunk_id`, `start_ts`, `end_ts`, `modality`, `source_uri`
- visual: `frame_count`, `preview_frame_uri?`
- audio: `transcript_snippet`, `language?` (embed transcripts with CLIP text to keep 512-d; MiniLM can be used offline for re-ranking, but not stored in Pinecone)
- faces: `cluster_id`, `face_count`, `thumbnail_uri?`

### Create/connect indices
```python
from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"]) 

def ensure_index(name: str, dimension: int, metric: str = "cosine"):
    names = [i["name"] for i in pc.list_indexes()]
    if name not in names:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

ensure_index(os.environ.get("PINECONE_INDEX_CHUNKS", "clip-chunks-v1"), 512, "cosine")
# ensure_index(os.environ.get("PINECONE_INDEX_FACES", "faces-v1"), <face_dim>, "cosine")
```

### Upsert visual chunk vectors (averaged CLIP image)
```python
import numpy as np
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"]) 
index = pc.Index(os.environ.get("PINECONE_INDEX_CHUNKS", "clip-chunks-v1"))

def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / (n + 1e-12)

def upsert_visual_chunk(video_id: str, chunk_id: str, start_ts: float, end_ts: float, vector_512: np.ndarray, metadata: dict):
    vid = f"chunk:{video_id}:{chunk_id}:visual"
    meta = {"video_id": video_id, "chunk_id": chunk_id, "modality": "visual", "start_ts": start_ts, "end_ts": end_ts}
    meta.update(metadata or {})
    index.upsert(vectors=[(vid, normalize(vector_512).tolist(), meta)])
```

If using `embeddings.py` frame averaging (CLIP): compute the per-chunk 512-d vector and call `upsert_visual_chunk` per chunk.

### Upsert audio transcript vectors (CLIP text)
```python
def upsert_audio_chunk(video_id: str, chunk_id: str, start_ts: float, end_ts: float, transcript: str, text_vec_512: np.ndarray, language: str = None):
    vid = f"chunk:{video_id}:{chunk_id}:audio"
    meta = {
        "video_id": video_id,
        "chunk_id": chunk_id,
        "modality": "audio",
        "start_ts": start_ts,
        "end_ts": end_ts,
        "transcript_snippet": transcript[:256],
    }
    if language:
        meta["language"] = language
    index.upsert(vectors=[(vid, normalize(text_vec_512).tolist(), meta)])
```

### Upsert face cluster vectors
```python
faces_index = pc.Index(os.environ.get("PINECONE_INDEX_FACES", "faces-v1"))

def upsert_face_cluster(video_id: str, cluster_id: str, face_vec: np.ndarray, metadata: dict):
    vid = f"face:{video_id}:{cluster_id}"
    meta = {"video_id": video_id, "cluster_id": cluster_id}
    meta.update(metadata or {})
    faces_index.upsert(vectors=[(vid, normalize(face_vec).tolist(), meta)])
```

### Batching
Prefer batching 100–500 vectors per call:
```python
def upsert_batch(items: list[tuple[str, np.ndarray, dict]], index_name: str):
    idx = pc.Index(index_name)
    payload = [(i, normalize(v).tolist(), m) for (i, v, m) in items]
    idx.upsert(vectors=payload)
```

### Completion criteria
- Indices exist and are healthy.
- A meaningful number of chunk and/or face vectors have been upserted with metadata.
- You can see counts via `pc.describe_index` or by querying with a test vector.


