# ClipABit Backend

Video processing backend that runs on Modal. Accepts video uploads via FastAPI and processes them in serverless containers.

## Quick Start

```bash
# 1. Install dependencies (creates .venv automatically)
uv sync

# 2. Authenticate with Modal (first time only - opens browser)
uv run modal token new

# 3. Configure Modal secrets (dev and prod)
modal secret create dev \
  ENVIRONMENT=dev \
  PINECONE_API_KEY=your_pinecone_api_key \
  R2_ACCOUNT_ID=your_r2_account_id \
  R2_ACCESS_KEY_ID=your_r2_access_key_id \
  R2_SECRET_ACCESS_KEY=your_r2_secret_access_key

modal secret create prod \
  ENVIRONMENT=prod \
  PINECONE_API_KEY=your_pinecone_api_key \
  R2_ACCOUNT_ID=your_r2_account_id \
  R2_ACCESS_KEY_ID=your_r2_access_key_id \
  R2_SECRET_ACCESS_KEY=your_r2_secret_access_key

# 4. Start dev server (hot-reloads on file changes, uses "dev" secret)
uv run dev
```

Note: `uv run` automatically uses the virtual environment - no need to activate it manually.

## How It Works

- `main.py` defines a Modal App with a `Server` class
- `/upload` endpoint accepts video files and spawns background processing jobs
- Environment variables stored in Modal secrets (no .env files needed)
- `uv run dev` automatically uses "dev" secret for development
- Production deployment handled via CI/CD or direct Modal CLI

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
