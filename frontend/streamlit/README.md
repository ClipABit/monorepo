# ClipABit Web Client

Streamlit interface for ClipABit technical demo. Supports uploading and searching through videos in a demo repository

## Quick Start

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Start the app
uv run streamlit run app.py
```

Opens at `http://localhost:8501`. Make sure the backend is running first.

## What It Does

- Let's you view all the videos in the demo repository
- Semantically search through those videos using the search field (Try it out!)
- Upload your own videos to get processed with our embedding engine and become available to search through
