import io
import time
import requests
import streamlit as st


st.set_page_config(page_title="ClipABit - Semantic Video Search", layout="centered")

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# API Configuration
BACKEND_API_URL = "https://clipabit01--clipabit-server-upload-dev.modal.run"
SEARCH_API_URL = "https://clipabit01--clipabit-server-search-dev.modal.run"


def upload_file_to_backend(api_url: str, file_bytes: bytes, filename: str, content_type: str | None = None):
    """Upload file to backend via multipart form-data."""
    files = {"file": (filename, io.BytesIO(file_bytes), content_type or "application/octet-stream")}
    resp = requests.post(api_url, files=files, timeout=300)
    return resp


def poll_job_status(status_url: str, job_id: str, max_wait: int = 120, poll_interval: int = 2):
    """Poll job status until complete or timeout."""
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{status_url}?job_id={job_id}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "unknown")

                if status in ["completed", "failed"]:
                    return data

            time.sleep(poll_interval)
        except requests.RequestException:
            time.sleep(poll_interval)

    return {"job_id": job_id, "status": "timeout", "message": "Job polling timed out"}


def search_videos(query: str):
    """Search for videos using the backend API."""
    try:
        # Backend expects GET request with query parameter
        resp = requests.get(
            SEARCH_API_URL,
            params={"query": query},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"Search failed with status {resp.status_code}: {resp.text}"}
    except requests.RequestException as e:
        return {"error": f"Search request failed: {str(e)}"}


@st.dialog("Upload Video", width="large")
def upload_dialog():
    """Modal dialog for video upload."""
    st.write("Upload a video file to process and index for searching.")
    
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"])
    
    if uploaded is not None:
        uploaded_bytes = uploaded.read()
        
        # Show video preview
        try:
            st.video(io.BytesIO(uploaded_bytes))
        except Exception:
            st.info("Preview not available for this format — proceeding to upload.")
        
        # File metadata
        st.write(f"**Filename:** {uploaded.name}")
        st.write(f"**Size:** {len(uploaded_bytes):,} bytes ({len(uploaded_bytes) / 1024 / 1024:.2f} MB)")
        
        # Upload button
        if st.button("Upload to Backend", type="primary", use_container_width=True):
            with st.spinner("Uploading video to backend..."):
                try:
                    resp = upload_file_to_backend(BACKEND_API_URL, uploaded_bytes, uploaded.name, uploaded.type)

                    if resp.status_code == 200:
                        try:
                            upload_data = resp.json()

                            if upload_data.get("status") == "processing":
                                job_id = upload_data.get("job_id")
                                st.info(f"Video uploaded successfully. Job ID: {job_id}")

                                status_url = BACKEND_API_URL.replace("-upload-", "-status-")

                                with st.spinner("Processing video... This may take a minute."):
                                    data = poll_job_status(status_url, job_id, max_wait=120, poll_interval=2)
                            else:
                                data = upload_data

                            # Display results
                            if data.get("status") == "completed":
                                st.success("Processing complete!")
                            elif data.get("status") == "failed":
                                st.error(f"Processing failed: {data.get('error', 'Unknown error')}")
                            elif data.get("status") == "timeout":
                                st.warning("Processing timed out. Job may still be running.")
                            else:
                                st.info(f"Status: {data.get('status', 'unknown')}")

                            # Display processing summary
                            if isinstance(data, dict):
                                st.subheader("Processing Summary")

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Status", data.get("status", "unknown"))
                                with col2:
                                    st.metric("Chunks", data.get("chunks", 0))
                                with col3:
                                    st.metric("Total Frames", data.get("total_frames", 0))
                                with col4:
                                    st.metric("Memory", f"{data.get('total_memory_mb', 0):.1f} MB")

                                st.code(f"Job ID: {data.get('job_id', 'N/A')}", language=None)

                                with st.expander("View raw response"):
                                    st.json(data)

                                if "chunk_details" in data and data["chunk_details"]:
                                    st.subheader("Chunk Details")
                                    for i, chunk in enumerate(data["chunk_details"], 1):
                                        with st.expander(f"Chunk {i}: {chunk.get('chunk_id', 'unknown')}"):
                                            meta = chunk.get('metadata', {})
                                            time_range = meta.get('timestamp_range', [0, 0])

                                            st.write(f"**Time Range:** {time_range[0]:.1f}s - {time_range[1]:.1f}s")
                                            st.write(f"**Duration:** {meta.get('duration', 0):.1f}s")
                                            st.write(f"**Frames:** {meta.get('frame_count', 0)} at {meta.get('sampling_fps', 0):.2f} fps")
                                            st.write(f"**Memory:** {chunk.get('memory_mb', 0):.2f} MB")
                                            st.write(f"**Complexity:** {meta.get('complexity_score', 0):.3f}")
                            else:
                                st.json(data)
                        except ValueError:
                            st.text(resp.text)
                    else:
                        st.error(f"Upload failed with status {resp.status_code}")
                        st.text(resp.text)

                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")


# Main UI
col_title, col_stats = st.columns([3, 1])
with col_title:
    st.title("🎬 ClipABit")
    st.subheader("Semantic Video Search")
with col_stats:
    # Show database stats if we have search results
    if st.session_state.search_results and 'stats' in st.session_state.search_results:
        stats = st.session_state.search_results['stats']
        st.metric("Vectors in DB", f"{stats.get('namespace_vectors', 0):,}")

# Header with search and upload button
col1, col2 = st.columns([5, 1])

with col1:
    search_query = st.text_input(
        "Search for video content",
        placeholder="e.g., 'a woman walking on a train platform'",
        label_visibility="collapsed"
    )

with col2:
    if st.button("📤 Upload", use_container_width=True):
        upload_dialog()

# Search button
if st.button("🔍 Search", type="primary", use_container_width=False):
    if search_query:
        with st.spinner("Searching..."):
            results = search_videos(search_query)
            st.session_state.search_results = results
    else:
        st.warning("Please enter a search query")

# Display results
if st.session_state.search_results:
    st.markdown("---")
    
    st.subheader("Search Results")
    
    results = st.session_state.search_results
    
    if "error" in results:
        st.error(f"Error: {results['error']}")
    else:
        # Display query and status
        if "query" in results:
            st.info(f"Query: {results['query']}")
        if "status" in results:
            st.success(f"Status: {results['status']}")
        
        # Show raw response for now since backend doesn't return actual results yet
        with st.expander("View API response", expanded=True):
            st.json(results)
        
        # Note to user about implementation
        st.info("ℹ️ Search endpoint is working! The backend currently returns a placeholder response. Once you implement the actual search logic in your Modal backend, results will appear here automatically.")

st.caption("ClipABit - Powered by CLIP embeddings and semantic search")