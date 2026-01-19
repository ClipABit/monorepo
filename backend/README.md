# ClipABit Backend

Video processing backend that runs on Modal. Built as a microservices architecture with three specialized apps for optimized cold start times.

## Architecture

The backend is split into three Modal apps:

| App | Purpose | Dependencies |
|-----|---------|--------------|
| **server** | API gateway, handles requests, lightweight operations | Minimal (FastAPI, boto3) |
| **search** | Semantic search with CLIP text encoder | Medium (torch, transformers) |
| **processing** | Video processing, embedding generation | Heavy (ffmpeg, opencv, torch, transformers) |

This architecture ensures that lightweight API calls (health, status, list) don't need to load heavy ML models.

## Quick Start

```bash
# 1. Install dependencies (creates .venv automatically)
uv sync

# 2. Authenticate with Modal (first time only - opens browser)
uv run modal token new

# 3. Start dev server
uv run serve-all
```

Note: `uv run` automatically uses the virtual environment - no need to activate it manually.

## Development CLI

| Command | Description |
|---------|-------------|
| `uv run serve-all` | Run all 3 apps in one terminal (color-coded logs, Ctrl+C stops all) |
| `uv run serve-server` | Run just the API server |
| `uv run serve-search` | Run just the search app |
| `uv run serve-processing` | Run just the processing app |

You can also run apps directly with Modal:

```bash
uv run modal serve apps/server.py
uv run modal serve apps/search_app.py
uv run modal serve apps/processing_app.py
```

## How It Works

- `apps/server.py` - API gateway, delegates heavy work to other apps via `modal.Function.lookup()`
- `apps/search_app.py` - Handles semantic search queries with CLIP text encoder
- `apps/processing_app.py` - Processes video uploads (chunking, embedding, storage)
- Environment variables stored in Modal secrets (no .env files needed)
- Cross-app communication uses Modal's `Function.lookup()` and `spawn()`

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

Note: Some integration tests require `ffmpeg` to be installed locally. Install with `brew install ffmpeg` on macOS.

## Deployment

Deployment is handled via GitHub Actions CI/CD. Apps are deployed in order:

1. `processing_app.py` (heavy dependencies)
2. `search_app.py` (medium dependencies)
3. `server.py` (API gateway - depends on the other two)

Manual deployment:

```bash
uv run modal deploy apps/processing_app.py
uv run modal deploy apps/search_app.py
uv run modal deploy apps/server.py
```
