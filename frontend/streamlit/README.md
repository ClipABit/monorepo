# ClipABit Web Client

Simple Streamlit interface for uploading videos to the backend. That's it.

## Quick Start

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Start the app
uv run streamlit run app.py
```

Opens at `http://localhost:8501`. Make sure the backend is running first.

## What It Does

- Lets you paste in your Modal backend URL (or uses the default)
- Upload a video file
- Shows you the response from the backend

The app sends a multipart POST request with the video as a `file` field. Backend handles the rest.
