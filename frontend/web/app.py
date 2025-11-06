"""Small Streamlit frontend for ClipABit.

Usage: run `streamlit run app.py` in this directory. The app lets you upload a video file
and sends it to your Modal backend API.

The backend must accept multipart/form-data with a `file` field for uploads.

This file is intentionally self-contained and uses only `requests` and `streamlit`.
"""

import base64
import io
from typing import Optional
import requests
import streamlit as st


st.set_page_config(page_title="ClipABit - Upload & Search", layout="centered")

st.title("ClipABit — Upload & Search")
st.caption("Upload a video to your Modal backend and run semantic search over stored chunks.")

api_url = st.text_input(
    "Upload Endpoint URL",
    value="https://clipabit01--clipabit-server-upload-dev.modal.run",
    help="Full URL of the Modal backend endpoint that accepts multipart file uploads.",
)

search_url = st.text_input(
    "Search Endpoint URL",
    value="",
    help="Full URL to your Modal /search endpoint (e.g., https://<your-modal-app>.modal.run/search)",
)


def upload_file_to_backend(api_url: str, file_bytes: bytes, filename: str, content_type: Optional[str] = None):
    """Upload file to backend via multipart form-data."""
    files = {"file": (filename, io.BytesIO(file_bytes), content_type or "application/octet-stream")}
    resp = requests.post(api_url, files=files, timeout=300)
    return resp


# File uploader (accept mp4 video or mp3 audio)
uploaded = st.file_uploader(
    "Choose a media file (mp4 or mp3)",
    type=["mp4", "mp3", "mov", "avi", "mkv", "webm"],
)

if uploaded is not None:
    # Read bytes once so we can both preview and upload
    uploaded_bytes = uploaded.read()
    
    # Show media preview (video or audio)
    try:
        if uploaded.type and uploaded.type.startswith("audio"):
            st.audio(io.BytesIO(uploaded_bytes))
        else:
            st.video(io.BytesIO(uploaded_bytes))
    except Exception:
        st.info("Preview not available for this format — proceeding to upload.")
    
    # File metadata
    st.write(f"**Filename:** {uploaded.name}")
    st.write(f"**Size:** {len(uploaded_bytes):,} bytes ({len(uploaded_bytes) / 1024 / 1024:.2f} MB)")
    
    # Upload button
    if st.button("Upload to Backend", type="primary"):
        if not api_url:
            st.error("Please set the backend API URL first.")
        else:
            with st.spinner("Uploading video to backend..."):
                try:
                    resp = upload_file_to_backend(api_url, uploaded_bytes, uploaded.name, uploaded.type)
                    
                    # Handle response
                    if resp.status_code == 200:
                        st.success("✅ Upload successful!")
                        try:
                            data = resp.json()
                            st.json(data)
                            
                            # Display thumbnail if backend returns one
                            if isinstance(data, dict) and "thumbnail_base64" in data:
                                try:
                                    thumb_b = base64.b64decode(data["thumbnail_base64"])
                                    st.image(io.BytesIO(thumb_b), caption="Thumbnail from backend")
                                except Exception:
                                    pass
                        except ValueError:
                            st.text(resp.text)
                    else:
                        st.error(f"Upload failed with status {resp.status_code}")
                        st.text(resp.text)
                        
                except requests.RequestException as e:
                    st.error(f"❌ Upload failed: {e}")


st.markdown("---")
st.subheader("Semantic Search")
st.caption("Query the Pinecone-backed index via your Modal backend.")

col1, col2 = st.columns(2)
with col1:
    query = st.text_input("Query text", value="")
with col2:
    st.markdown("\u00A0")
    st.markdown("\u00A0")
    run_search = st.button("Search", type="primary")

video_id_filter = st.text_input("Filter: video_id (optional)", value="")

def _build_filters():
    f = {}
    if video_id_filter.strip():
        f["video_id"] = {"$eq": video_id_filter.strip()}
    return f or None

def _post_json(url: str, payload: dict):
    # Increased timeout to handle first-call cold starts (model downloads)
    return requests.post(url, json=payload, timeout=180)

if run_search:
    if not search_url:
        st.error("Please set the Search Endpoint URL.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        payload = {"query": query.strip(), "top_k": 3, "filters": _build_filters()}
        with st.spinner("Searching..."):
            try:
                resp = _post_json(search_url, payload)
                if resp.status_code == 200:
                    data = resp.json()
                    results = (data or {}).get("results") or []
                    if not results:
                        st.info("No results")
                    for r in results:
                        with st.container(border=True):
                            st.write({
                                k: v for k, v in r.items()
                                if k in ["id", "score", "video_id", "chunk_id", "modality", "start_ts", "end_ts", "transcript_snippet"] and v is not None
                            })
                            if r.get("preview_frame_uri"):
                                try:
                                    st.image(r["preview_frame_uri"], caption="preview_frame_uri")
                                except Exception:
                                    pass
                else:
                    st.error(f"Search failed with status {resp.status_code}")
                    try:
                        st.text(resp.text)
                    except Exception:
                        pass
            except requests.RequestException as e:
                st.error(f"❌ Search failed: {e}")

