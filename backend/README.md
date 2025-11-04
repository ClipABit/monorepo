# ClipABit Backend

Video processing backend that runs on Modal. Accepts video uploads via FastAPI and processes them in serverless containers.

## Quick Start

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Authenticate with Modal (first time only - opens browser)
uv run modal token new

# Start dev server (hot-reloads on file changes)
uv run modal serve main.py
```

Note: `uv run` automatically uses the virtual environment - no need to activate it manually.

## How It Works

- `main.py` defines a Modal App with a `Server` class
- `/upload` endpoint accepts video files and spawns background processing jobs

[...]

## Managing Dependencies

```bash
uv add package-name              # Add new dependency
uv add --dev package-name        # Add dev dependency
uv remove package-name           # Remove dependency
uv sync --upgrade                # Update all packages
```
