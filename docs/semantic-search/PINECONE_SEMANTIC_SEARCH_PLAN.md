## Pinecone Semantic Search Integration Plan

### Why
- Leverage Pinecone’s native vector search to retrieve the most relevant video chunks across modalities (visual frames → CLIP image embeddings, audio → CLIP text embeddings from transcript, faces → face embeddings/clusters).
- Standardize retrieval so a query embedded with CLIP can return top-k results with metadata (timestamps, chunk ids, modalities, clusters) similar to the prior Multimodal-Audio-Search demo.

### Scope
- Use CLIP (openai/clip-vit-base-patch32) for text and image embeddings now; audio uses transcript text embeddings.
- Pinecone will store embeddings for chunks and faces with searchable metadata.
- Expose a search endpoint that embeds a query via CLIP text encoder, queries Pinecone, and returns top-k hits with rich details.

## Architecture Overview
- Ingestion
  - Existing pipeline: video → chunk → frames → CLIP image encodings → average to a single chunk-level vector.
  - Audio: transcribe to text; embed text with CLIP text encoder.
  - Faces: detect/cluster; store a representative vector per face cluster.
  - Upsert all vectors into Pinecone with consistent metadata.
- Retrieval
  - Query → embed via CLIP text encoder → Pinecone `query` (cosine) over the relevant index(es) → rank → map to response shape (include timestamps, chunk ids, preview URIs, etc.).

## Index Design
- Index type: Pinecone serverless index with cosine metric.
- Dimensions:
  - CLIP ViT-B/32 text and image embeddings: 512 dims.
  - Face embeddings (DeepFace or equivalent): depends on model (commonly 128–512). Use a separate index if dims differ.
- Proposed indices
  - clip-chunks-v1 (dim=512, metric=cosine) → multimodal chunk vectors (visual averaged + transcript text).
  - faces-v1 (dim=<face_dim>, metric=cosine) → representative vectors for face clusters.
- Upsert id convention
  - chunk: `chunk:{video_id}:{chunk_id}:{modality}` where modality ∈ {visual, audio, combined}.
  - face: `face:{video_id}:{cluster_id}`.
- Metadata (for filtering/ranking)
  - Common: `video_id`, `chunk_id`, `start_ts`, `end_ts`, `source_uri`, `modality`.
  - Visual: `frame_count`, optional `preview_frame_uri`.
  - Audio: `transcript_snippet` (short), `language`.
  - Faces: `cluster_id`, `face_count`, optional `thumbnail_uri`.

## Data Flow
1. Processing produces:
   - Visual: per-chunk averaged CLIP image vector (512).
   - Audio: transcript text → CLIP text vector (512).
   - Faces: per-cluster vector (face_dim).
2. Upsert
   - Build Pinecone client from env.
   - Ensure index exists; create if missing.
   - Batch upsert with ids + vectors + metadata.
3. Query
   - Input query string → CLIP text embedding.
   - Pinecone `query` topK over `clip-chunks-v1` (and `faces-v1` if searching faces using the same query flow or a separate UI toggle).
   - Map to response with details used in Multimodal-Audio-Search.

## Environment & Config
- Env vars
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_CHUNKS=clip-chunks-v1`
  - `PINECONE_INDEX_FACES=faces-v1`
  - Optional: `PINECONE_CLOUD`, `PINECONE_REGION` (serverless), or `PINECONE_HOST` (as needed by client).
- Dependencies (backend)
  - `pinecone-client` (v3+), `torch`, `transformers`, `opencv-python-headless`, `ffmpeg-python`, `numpy`, `python-dotenv`.

## Components To Add
1. Pinecone client module (extend `backend/pinecone_connect.py`)
   - Ensure connect/create index helpers for both indices.
   - Add `query_index(index_name, vector, top_k, filter)` helper.
2. CLIP embedding service
   - Text: CLIP text encoder.
   - Image: CLIP image encoder.
   - Normalization: L2-normalize vectors before upsert/query (cosine friendly).
3. Ingestion pipeline hooks
   - After chunk processing completes, produce vectors and upsert to Pinecone.
   - Audio transcript → embed → upsert (modality=audio).
   - Faces → per-cluster representative vector → upsert to faces index.
4. Search API endpoint
   - Input: `{ query: string, top_k?: number, modalities?: ["visual","audio","faces"], filters?: object }`.
   - Flow: embed query → query Pinecone → map results → return consistent shape.
5. Result mapping
   - Shape mirrors Multimodal-Audio-Search: include `chunk_id`, `video_id`, `score`, `start_ts`, `end_ts`, `modality`, `transcript_snippet`, `preview_frame_uri`, and any cluster details for faces.

## Implementation Steps
1. Pinecone wiring
   - Extend `pinecone_connect.py` with:
     - `connect_to_pinecone()` (already present)
     - `ensure_index(index_name, dimension, metric="cosine")`
     - `upsert_vectors(index_name, items)` where items = `[(id, vector, metadata), ...]`
     - `query_vectors(index_name, vector, top_k=10, filter=None)`
2. CLIP embeddings
   - Add a `clip_service.py` providing:
     - `embed_text(query: str) -> np.ndarray[512]`
     - `embed_image(np.ndarray | PIL.Image) -> np.ndarray[512]`
     - Utility: `normalize(vec)`.
3. Ingestion integration
   - In `backend/main.py` processing flow (Modal background task), after chunking:
     - Compute CLIP image embeddings per frame → average per chunk → upsert to `clip-chunks-v1` with `modality=visual`.
     - If transcript available: embed text → upsert with `modality=audio` and `transcript_snippet`.
     - Faces: when clustering completes, upsert representative face vectors into `faces-v1`.
4. Search endpoint (FastAPI via Modal)
   - Add `@modal.fastapi_endpoint(method="POST")` `/search`:
     - Body: `{ query: string, top_k?: number, modalities?: string[], filters?: object }`.
     - Steps: embed → query indices per modality → merge/rank → map fields → return.
5. Backfill
   - Script `backend/scripts/backfill_pinecone.py` to traverse existing stored chunks/faces and upsert.
   - Optional notebook for verification/quality checks.
6. Tests
   - Unit tests for embedding shapes, normalization, and metadata mapping.
   - E2E smoke: ingest a small sample, run /search, assert at least one non-empty result.

## API (Proposed)
- Request
```json
{
  "query": "person smiling with red hoodie", 
  "top_k": 10,
  "modalities": ["visual", "audio"],
  "filters": {"video_id": "abc123"}
}
```
- Response
```json
{
  "results": [
    {
      "id": "chunk:abc123:42:visual",
      "score": 0.8123,
      "video_id": "abc123",
      "chunk_id": "42",
      "modality": "visual",
      "start_ts": 12.4,
      "end_ts": 18.9,
      "preview_frame_uri": "s3://.../abc123/42.jpg"
    },
    {
      "id": "chunk:abc123:43:audio",
      "score": 0.7851,
      "video_id": "abc123",
      "chunk_id": "43",
      "modality": "audio",
      "start_ts": 19.0,
      "end_ts": 26.0,
      "transcript_snippet": "… we discuss the red hoodie …"
    }
  ]
}
```

## Operational Notes
- Normalization: ensure consistent vector normalization across upsert and query.
- Batching: upsert in batches (e.g., 100–500 vectors) for throughput.
- Filters: use Pinecone metadata filters (e.g., `{ "video_id": {"$eq": "abc123"} }`).
- Idempotency: deterministic ids allow safe re-upserts.
- Costs: prefer serverless; keep index counts and dimensions minimal.
- Observability: log query latency, top-k hit scores, and cache model(s) once per container.

## Milestones
1. Wiring + indices created (env + ensure_index)
2. CLIP service ready (text/image embed)
3. Ingestion writing vectors (visual + audio)
4. Search endpoint returning top-k with mapped fields
5. Backfill existing data
6. Tests + minimal dashboard/metrics

## Next Actions
- Confirm face embedding dimension and model → finalize `faces-v1` index.
- Add index creation on startup with correct dimensions.
- Implement search API and verify with a small test video.


