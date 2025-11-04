## Step 03 — Integrate Streamlit-style Search Logic with Pinecone

Goal: adapt your Streamlit demo’s search logic to use Pinecone as the vector store, using CLIP text embeddings for queries and returning top-k results with rich metadata.

Note: I don’t have direct access to your Streamlit demo code in this workspace. Below is a drop-in structure that mirrors common Streamlit patterns and maps to Pinecone queries. Replace UI bits with your actual Streamlit components and reuse your display/result mapping logic.

### Core search function
```python
import os
import numpy as np
from pinecone import Pinecone
from embeddings import generate_clip_text_embedding

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"]) 
INDEX_NAME = os.environ.get("PINECONE_INDEX_CHUNKS", "clip-chunks-v1")
index = pc.Index(INDEX_NAME)

def run_semantic_search(query: str, top_k: int = 10, filters: dict | None = None):
    q = generate_clip_text_embedding(query).numpy()  # normalized [512]
    res = index.query(
        vector=q.tolist(),
        top_k=top_k,
        filter=filters,
        include_metadata=True,
    )
    # Map Pinecone hits to your prior display schema
    results = []
    for m in res.matches:
        meta = m.metadata or {}
        results.append({
            "id": m.id,
            "score": m.score,
            "video_id": meta.get("video_id"),
            "chunk_id": meta.get("chunk_id"),
            "modality": meta.get("modality"),
            "start_ts": meta.get("start_ts"),
            "end_ts": meta.get("end_ts"),
            "preview_frame_uri": meta.get("preview_frame_uri"),
            "transcript_snippet": meta.get("transcript_snippet"),
        })
    return results
```

### Streamlit wiring (example shell)
```python
# streamlit_app.py (example structure)
import streamlit as st
from search_core import run_semantic_search  # import the function above

st.title("Multimodal Semantic Search")
query = st.text_input("Describe what you’re looking for")
top_k = st.slider("Top-K", 1, 50, 10)

if st.button("Search") and query:
    results = run_semantic_search(query, top_k=top_k)
    for r in results:
        st.write({k: v for k, v in r.items() if v is not None})
```

### Optional: FastAPI/Modal endpoint for the web client
You can expose the same logic via your existing Modal FastAPI endpoints (see `backend/main.py`).

```python
# inside your Modal Server class
from fastapi import Body

@modal.fastapi_endpoint(method="POST")
async def search(self, payload: dict = Body(...)):
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 10))
    filters = payload.get("filters")
    results = run_semantic_search(query, top_k=top_k, filters=filters)
    return {"results": results}
```

### Filters and modalities
- To limit by modality, add a metadata filter: `{ "modality": {"$in": ["visual", "audio"] } }`
- To limit to one video: `{ "video_id": {"$eq": "abc123"} }`

### Validation checklist
- Query vectors are 512-d normalized CLIP text embeddings.
- Pinecone index metric is `cosine`.
- Results include the metadata fields your UI expects (timestamps, modality, etc.).

### Modal integration and run commands
- Backend (Terminal 1):
  - `cd backend`
  - `uv sync`
  - `uv run modal serve main.py`
- Frontend (Terminal 2):
  - `cd frontend/web`
  - `uv sync`
  - `uv run streamlit run app.py`

Once the Modal server is live, add a `/search` endpoint (as above) in `backend/main.py` and call it from Streamlit. For your current scale, perform one query at a time with small `top_k`.


