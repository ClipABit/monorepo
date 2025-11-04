## Step 02 — Embed the Query with CLIP and (Optionally) Store It

Goal: take a user query string, embed it via CLIP text encoder, and use that vector to query Pinecone. Optionally, store query vectors for analytics/debugging.

### Prereqs
- Step 01 completed (indices exist and contain vectors)
- `embeddings.py` available (functions like `generate_clip_text_embedding`)
- Env: `PINECONE_API_KEY`, `PINECONE_INDEX_CHUNKS`

### Generate CLIP text embedding
Using your provided `embeddings.py` logic (CLIP ViT-B/32):
```python
import numpy as np
from embeddings import generate_clip_text_embedding

def text_to_vec(query: str) -> np.ndarray:
    # generate_clip_text_embedding returns a normalized torch.Tensor [512]
    t = generate_clip_text_embedding(query)
    return t.numpy()
```

### Query Pinecone (semantic search)
```python
from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"]) 
index = pc.Index(os.environ.get("PINECONE_INDEX_CHUNKS", "clip-chunks-v1"))

def search_chunks(query_vec_512: np.ndarray, top_k: int = 10, filters: dict | None = None):
    res = index.query(
        vector=query_vec_512.tolist(),
        top_k=top_k,
        filter=filters,
        include_metadata=True,
    )
    return res.matches  # [{id, score, metadata, ...}]
```

### Optional: store query vectors
You usually don’t need to store queries; Pinecone compares your query vector to stored chunk vectors on the fly. If you want to log queries for reproducibility/analytics, either:

- Use a separate index like `queries-v1` (dim=512), or
- Store inside `clip-chunks-v1` under a distinct `namespace="queries"`.

```python
def store_query(query_id: str, query_text: str, query_vec_512: np.ndarray, namespace: str = "queries"):
    index.upsert(
        vectors=[(f"query:{query_id}", query_vec_512.tolist(), {"text": query_text})],
        namespace=namespace,
    )
```

### End-to-end helper
```python
def embed_and_search(query: str, top_k: int = 10, filters: dict | None = None):
    qv = text_to_vec(query)  # 512-d normalized
    matches = search_chunks(qv, top_k=top_k, filters=filters)
    # shape results for UI
    return [
        {
            "id": m.id,
            "score": m.score,
            **(m.metadata or {})
        }
        for m in matches
    ]
```

### Notes
- Ensure vectors are normalized in your `embeddings.py` (they are), and index metric is `cosine`.
- Use metadata filters to scope search by `video_id`, `modality`, time ranges, etc.
- If you must persist queries, prefer a separate index or a dedicated namespace to avoid mixing with chunks.


