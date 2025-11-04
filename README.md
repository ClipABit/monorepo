# ClipABit

Video processing app split into backend and frontend.

- `/backend` - Modal-hosted FastAPI service that processes videos
- `/frontend/web` - Streamlit uploader UI
- `/frontend/plugin` - DaVinci Resolve plugin (coming soon)

## Running Locally

Open two terminals:

```bash
# Terminal 1: Backend
cd backend
uv sync
uv run modal serve main.py

# Terminal 2: Frontend
cd frontend/web
uv sync
uv run streamlit run app.py
```

Backend runs on Modal, frontend opens at `localhost:8501`.
