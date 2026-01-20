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
- No cross-app lookups needed - services call each other directly within the same app
- Uses `api/fastapi_router_dev.py` which accepts worker class references instead of doing lookups
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

## Deployment

Deployment is handled via GitHub Actions CI/CD. **Only the individual apps are deployed** (not dev_combined.py):

1. `processing_app.py` (heavy dependencies)
2. `search_app.py` (medium dependencies)
3. `server.py` (API gateway - depends on the other two)
