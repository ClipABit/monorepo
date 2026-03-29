# ClipABit Backend

Video processing backend that runs on Modal. Built as a microservices architecture with three specialized apps for optimized cold start times.

## Architecture

The backend uses different architectures for development vs production:

### Production (staging/prod)

Split into three Modal apps for optimal cold start times:

| App | Purpose | Dependencies |
|-----|---------|--------------|
| **server** | API gateway, handles requests, lightweight operations | Minimal (FastAPI, boto3) |
| **search** | Semantic search with CLIP text encoder | Medium (torch, transformers) |
| **processing** | Video processing, embedding generation | Heavy (ffmpeg, opencv, torch, transformers) |

This architecture ensures that lightweight API calls (health, status, list) don't need to load heavy ML models.

### Local Development (dev)

Single combined Modal app (`dev_combined.py`) with all three services:

| App | Purpose |
|-----|---------|
| **dev-combined** | Server + Search + Processing in one app |
This allows hot-reload on all services without cross-app lookup issues. Cold start time is acceptable for local development where iteration speed matters more than cold start performance.

## Namespace Pool & Vector Quota System

### Shared Namespace Pool

Users are assigned to a shared pool of **20 Pinecone namespaces** (`ns_00` through `ns_19`). This replaces the old per-user namespace hashing.

| Constant | Value | Description |
|----------|-------|-------------|
| `NAMESPACE_POOL_SIZE` | 20 | Total namespaces in the pool |
| `MAX_VECTORS_PER_NAMESPACE` | 100,000 | Max vectors per namespace |
| `MAX_USERS_PER_NAMESPACE` | 10 | Max users sharing a namespace |
| `DEFAULT_VECTOR_QUOTA` | 1,000 | Per-user vector limit |

**Assignment strategy:** New users are assigned to the namespace with the most remaining vector capacity (even-spread). Once assigned, the binding is permanent.

**Legacy backfill:** Users with old-format namespaces (not starting with `ns_`) are automatically reassigned to the pool on their next authentication.

### Vector Quota Enforcement

Quota is enforced at **two levels** to prevent overspending:

1. **Soft pre-check (upload endpoint):** Before spawning processing, the server checks `user_store.check_quota()`. Returns HTTP 429 if the user is at or over their vector limit.

2. **Hard gate (processing service):** Before any Pinecone upserts, the processing service re-checks the quota. If `current_count + new_chunks > quota`, processing aborts and no vectors are written. This catches concurrent uploads that pass the soft check simultaneously.

After successful upserts, vector counts are atomically incremented at both the **user level** and **namespace level** in Firestore using `Increment()`.

### Firestore Data Model

```
users/{user_id}
  ├── user_id: string
  ├── namespace: string          # e.g. "ns_03"
  ├── vector_count: number       # current vectors stored
  ├── vector_quota: number       # max allowed (see DEFAULT_VECTOR_QUOTA)
  ├── created_at: string
  └── videos/{hashed_identifier}
        ├── hashed_identifier: string
        ├── chunk_count: number
        ├── filename: string
        └── created_at: string

namespaces/{ns_id}               # e.g. "ns_00"
  ├── namespace_id: string
  ├── vector_count: number       # total vectors across all users
  └── user_count: number         # users assigned to this namespace
```

### Search Isolation

Users sharing a namespace are isolated via **metadata filtering**. Every vector upserted to Pinecone includes `user_id` and optionally `project_id` in its metadata. Authenticated search automatically applies:

```json
{"user_id": {"$eq": "<authenticated_user_id>"}}
```

The public demo search endpoint (`/demo-search`) uses a hardcoded `web-demo` namespace with no user filter.

## API Endpoints

### Server Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check |
| `GET` | `/status?job_id=` | Yes | Poll job processing status |
| `GET` | `/quota` | Yes | Get current vector usage and quota |
| `POST` | `/upload` | Yes | Upload video(s) for processing |
| `GET` | `/videos` | Yes | List videos in user's namespace |
| `POST` | `/cache/clear` | Yes | Clear URL cache for user's namespace |

### Search Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Search service health check |
| `GET` | `/search?query=&top_k=` | Yes | Semantic search (user's namespace, filtered by user_id) |
| `GET` | `/demo-search?query=&top_k=` | No | Public demo search (rate limited, `web-demo` namespace) |

### Upload Request

```
POST /upload
Content-Type: multipart/form-data
Authorization: Bearer <token>

Form fields:
  files:              Video file(s) - supports single or batch (up to 200)
  namespace:          (ignored) Server always uses user's assigned namespace
  hashed_identifier:  Client-generated hash identifying the video file
  project_id:         Resolve project identifier for metadata filtering
```

### Upload Response (single)

```json
{
  "job_id": "uuid",
  "filename": "clip.mp4",
  "content_type": "video/mp4",
  "size_bytes": 1234567,
  "status": "processing",
  "namespace": "ns_03",
  "vector_count": 4821,
  "vector_quota": 1000
}
```

### Upload Response (batch)

```json
{
  "batch_job_id": "batch-uuid",
  "status": "processing",
  "total_submitted": 5,
  "total_videos": 5,
  "successfully_spawned": 5,
  "failed_validation": 0,
  "failed_at_upload": 0,
  "namespace": "ns_03",
  "vector_count": 4821,
  "vector_quota": 1000
}
```

### Quota Response

```
GET /quota
Authorization: Bearer <token>
```

```json
{
  "user_id": "auth0|abc123",
  "namespace": "ns_03",
  "vector_count": 4821,
  "vector_quota": 1000,
  "vectors_remaining": 179
}
```

### Vector Metadata (stored in Pinecone)

Each vector upserted to Pinecone includes this metadata:

```json
{
  "user_id": "auth0|abc123",
  "project_id": "proj_abc",
  "file_filename": "clip.mp4",
  "file_type": "video/mp4",
  "file_hashed_identifier": "client_hash_xyz",
  "start_time_s": 0.0,
  "end_time_s": 5.0,
  "frame_count": 8,
  "complexity_score": 0.42
}
```

### Processing Pipeline

1. **Preprocessing** — Video is chunked by scene detection, frames extracted with adaptive sampling
2. **Embedding** — CLIP vision encoder generates 512-dim embeddings per chunk
3. **Hard quota check** — Re-verifies `current_count + new_chunks <= quota` before any Pinecone writes
4. **Upsert** — Vectors written to Pinecone in user's assigned namespace with full metadata
5. **Quota update** — Atomic `Increment()` on both user and namespace Firestore docs
6. **Video registration** — Chunk count stored in `users/{id}/videos/{hash}` subcollection
7. **Rollback on failure** — If any upsert fails, all previously upserted vectors are deleted from Pinecone

## Quick Start

```bash
# 1. Install dependencies (creates .venv automatically)
uv sync

# 2. Authenticate with Modal (first time only - opens browser)
uv run modal token new

# 3. Start dev server
uv run dev
```

Note: `uv run` automatically uses the virtual environment - no need to activate it manually.

## Development CLI

### Local Development (Combined App)

For local development, use the combined app that includes all services in one:

| Command | Description |
|---------|-------------|
| `uv run dev` | Run combined dev app (server + search + processing in one) |

This uses `apps/dev_combined.py` which combines all three services into a single Modal app. Benefits:
- Hot-reload works on all services (server, search, processing)
- No cross-app lookup issues
- Easy to iterate on any part of the system

**Note:** Cold starts will be slower since all dependencies load together, but this is acceptable for local development where iteration speed matters more than cold start performance.

### Individual Apps (For Testing/Debugging)

You can also run individual apps if needed:

| Command | Description |
|---------|-------------|
| `uv run server` | Run just the API server |
| `uv run search` | Run just the search app |
| `uv run processing` | Run just the processing app |


**Note:** Cross-app communication only works between deployed apps, not ephemeral serve apps. For full system testing, use `uv run dev` (combined app) or deploy the apps.

## How It Works

### Production Architecture (staging/prod)

- `apps/server.py` - API gateway, delegates heavy work to other apps via `modal.Cls.from_name()`
- `apps/search_app.py` - Handles semantic search queries with CLIP text encoder
- `apps/processing_app.py` - Processes video uploads (chunking, embedding, storage)
- Cross-app communication uses Modal's `Cls.from_name()` for lookups and `spawn()`/`remote()` for calls
- Environment variables stored in Modal secrets (no .env files needed)

### Local Development Architecture (dev)

- `apps/dev_combined.py` - All three services in one app for easy iteration
- Uses `api/fastapi_router.py` (configured for dev combined mode) which accepts worker class references instead of doing lookups
- No cross-app lookups needed - services call each other directly within the same app
- Hot-reload works on all services simultaneously

## Managing Dependencies

```bash
uv add package-name              # Add new dependency
uv add --dev package-name        # Add dev dependency
uv remove package-name           # Remove dependency
uv sync --upgrade                # Update all packages
```

## Running Tests

```bash
uv run pytest                    # Run all tests
uv run pytest -v                 # Verbose output
uv run pytest --cov              # With coverage
```

Note: Some integration tests require `ffmpeg` to be installed locally.

### Key Test Files

| File | Coverage |
|------|----------|
| `test_processing_quota.py` | Hard quota enforcement, increment/decrement, rollback |
| `test_namespace_pool.py` | Quota rejection (429), multi-user namespace isolation, search filtering, metadata injection |
| `test_user_store_quota.py` | Namespace assignment, even-spread, quota checks, video registration |
| `test_user_store_connector.py` | User CRUD, backfill of legacy namespaces |
| `test_search_fastapi_router.py` | Search endpoint, auth, user namespace resolution |
| `test_search_namespace.py` | Search user isolation, demo search unchanged |

## Deployment

Deployment is handled via GitHub Actions CI/CD. **Only the individual apps are deployed** (not dev_combined.py):

1. `processing_app.py` (heavy dependencies)
2. `search_app.py` (medium dependencies)
3. `server.py` (API gateway - depends on the other two)
