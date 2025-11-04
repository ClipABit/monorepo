## Modal Runbook — Wiring Semantic Search into Modal + Streamlit

### Environment
- `.env` with:
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_CHUNKS=clip-chunks-v1`
  - `PINECONE_CLOUD=aws`
  - `PINECONE_REGION=us-east-1`
- One index only (512-d, cosine) for all modalities: visual, audio, face.
- For your scale (one chunk at a time), batch size can be `1`.

### Start services
- Backend (Terminal 1):
  - `cd backend`
  - `uv sync`
  - `uv run modal serve main.py`
- Frontend (Terminal 2):
  - `cd frontend/web`
  - `uv sync`
  - `uv run streamlit run app.py`

### Hook points in `backend/main.py`
- In `Server.process_video` (Modal background task):
  1) After frame extraction + CLIP image embedding average → call `upsert_visual_chunk(...)`.
  2) After transcription (Whisper base) → embed transcript with CLIP text → `upsert_audio_chunk(...)`.
  3) After face clustering → compute 512-d face representative → `upsert_face_cluster(...)`.
- Add a FastAPI route for search:
  - `@modal.fastapi_endpoint(method="POST")` `/search` → embed query with CLIP text → `query_vectors(...)` → return matches + metadata.

### Dimensionality and models
- CLIP ViT-B/32: 512-d for both text and image.
- Faces: use a 512-d embedding (ArcFace/Facenet512) for compatibility with the single index.
- MiniLM (384-d) is optional for re-ranking outside Pinecone. Don’t store MiniLM vectors in the 512-d index.

### Cost-conscious defaults
- Upsert only chunk-level vectors (no per-frame storage).
- Use `include_values=False`, `include_metadata=True` for queries.
- Keep `top_k` small (≤ 25) and use metadata filters.

### Smoke validation
- After uploading a small test video:
  - Verify upserts (by querying a CLIP text like the obvious content in the video).
  - Confirm IDs look like `chunk:{video_id}:{chunk_id}:{modality}` or `face:{video_id}:{cluster_id}`.
  - Check timestamps and URIs in metadata render correctly in Streamlit.


