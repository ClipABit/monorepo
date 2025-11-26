import io
import time
import requests
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="ClipABit - Semantic Video Search",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'show_upload' not in st.session_state:
    st.session_state.show_upload = False

# API endpoints
SEARCH_API_URL = "https://clipabit01--clipabit-server-search-dev.modal.run"
UPLOAD_API_URL = "https://clipabit01--clipabit-server-upload-dev.modal.run"
STATUS_API_URL = "https://clipabit01--clipabit-server-status-dev.modal.run"
GET_FRAME_API_URL = "https://clipabit01--clipabit-server-get-frame-dev.modal.run"


def search_videos(query: str):
    """Send search query to backend."""
    try:
        resp = requests.post(SEARCH_API_URL, json={"query": query, "top_k": 10}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"Search failed with status {resp.status_code}"}
    except requests.RequestException as e:
        return {"error": str(e)}


def upload_file_to_backend(file_bytes: bytes, filename: str, content_type: str | None = None):
    """Upload file to backend via multipart form-data."""
    files = {"file": (filename, io.BytesIO(file_bytes), content_type or "application/octet-stream")}
    resp = requests.post(UPLOAD_API_URL, files=files, timeout=300)
    return resp


def poll_job_status(job_id: str, max_wait: int = 120, status_placeholder=None):
    """Poll job status until complete or timeout."""
    start_time = time.time()
    poll_count = 0
    
    while time.time() - start_time < max_wait:
        poll_count += 1
        elapsed = int(time.time() - start_time)
        
        try:
            resp = requests.get(f"{STATUS_API_URL}?job_id={job_id}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "unknown")
                
                # Update status display
                if status_placeholder:
                    if status == "processing":
                        status_placeholder.info(f"⏳ Processing... ({elapsed}s elapsed, poll #{poll_count})")
                    elif status == "completed":
                        status_placeholder.success(f"✓ Processing complete! (took {elapsed}s)")
                    elif status == "failed":
                        status_placeholder.error(f"✗ Processing failed after {elapsed}s")
                
                if status in ["completed", "failed"]:
                    return data
            else:
                if status_placeholder:
                    status_placeholder.warning(f"⚠ Checking status... ({elapsed}s elapsed)")
            
            time.sleep(2)
        except requests.RequestException as e:
            if status_placeholder:
                status_placeholder.warning(f"⚠ Connection issue, retrying... ({elapsed}s elapsed)")
            time.sleep(2)
    
    return {"job_id": job_id, "status": "timeout", "message": "Job polling timed out"}


# Upload dialog
@st.dialog("Upload Video")
def upload_dialog():
    st.write("Upload a video to add it to the searchable database.")
    
    uploaded = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"])
    
    if uploaded is not None:
        uploaded_bytes = uploaded.read()
        
        # Show video preview
        try:
            st.video(io.BytesIO(uploaded_bytes))
        except Exception:
            st.info("Preview not available for this format.")
        
        # File metadata
        st.write(f"**Filename:** {uploaded.name}")
        st.write(f"**Size:** {len(uploaded_bytes):,} bytes ({len(uploaded_bytes) / 1024 / 1024:.2f} MB)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Upload", type="primary", use_container_width=True):
                try:
                    # Upload phase
                    with st.spinner("📤 Uploading video to server..."):
                        resp = upload_file_to_backend(uploaded_bytes, uploaded.name, uploaded.type)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "processing":
                            job_id = data.get("job_id")
                            st.success(f"✓ Video uploaded! Job ID: `{job_id}`")
                            
                            st.markdown("---")
                            st.subheader("Processing Status")
                            st.caption("This may take a few minutes depending on video length...")
                            
                            # Create placeholder for live status updates
                            status_placeholder = st.empty()
                            status_placeholder.info("⏳ Starting video processing...")
                            
                            # Poll for completion with live updates
                            result = poll_job_status(job_id, max_wait=120, status_placeholder=status_placeholder)
                            
                            st.markdown("---")
                            
                            if result.get("status") == "completed":
                                st.success("🎉 Video processing complete!")
                                
                                # Display results
                                st.subheader("Processing Results")
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Video Chunks", result.get("chunks", 0))
                                with col_b:
                                    st.metric("Frames Embedded", result.get("total_frames", 0))
                                with col_c:
                                    st.metric("Memory Used", f"{result.get('total_memory_mb', 0):.1f} MB")
                                
                                st.info("✨ Video frames are now searchable! Close this dialog and try searching for content.")
                                
                                with st.expander("📊 View detailed processing info"):
                                    st.json(result)
                                
                            elif result.get("status") == "failed":
                                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                                with st.expander("View error details"):
                                    st.json(result)
                            elif result.get("status") == "timeout":
                                st.warning("⏰ Processing timed out (120s limit reached). The job may still be running in the background.")
                                st.info(f"You can manually check status at: {STATUS_API_URL}?job_id={job_id}")
                            else:
                                st.warning(f"⚠ Unknown status: {result.get('status', 'unknown')}")
                                st.json(result)
                        else:
                            st.error("❌ Upload failed - unexpected response")
                            st.json(data)
                    else:
                        st.error(f"Upload failed with status {resp.status_code}")
                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


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
    
    # Header with toggle
    col_header, col_toggle = st.columns([3, 1])
    with col_header:
        st.subheader("Search Results")
    with col_toggle:
        show_frames = st.checkbox("Show frames", value=True, help="Fetch and display frame images (for testing/debugging)")
    
    results = st.session_state.search_results
    
    if "error" in results:
        st.error(f"Error: {results['error']}")
    else:
        # Display query echo
        if "query" in results:
            st.info(f"Query: {results['query']}")
        
        # Display results
        search_results = results.get('results', [])
        if search_results:
            for i, result in enumerate(search_results, 1):
                metadata = result.get('metadata', {})
                score = result.get('score', 0)
                result_id = result.get('id', '')
                with st.container():
                    st.markdown(f"### Result {i} - Score: {score:.3f}")
                    st.write(f"- Chunk ID: `{metadata.get('chunk_id', 'N/A')}`")
                    st.write(f"- Video ID: `{metadata.get('video_id', 'N/A')}`")
                    st.write(f"- Duration: **{metadata.get('duration', 0):.2f}s**")
                    st.write(f"- Start Time: {metadata.get('start_time_s', 0):.2f}s")
                    st.write(f"- End Time: {metadata.get('end_time_s', 0):.2f}s")
                    st.write(f"- Frame Count: {metadata.get('frame_count', 0)}")
                    st.write(f"- Complexity Score: {metadata.get('complexity_score', metadata.get('complexity', 0)):.3f}")
                    st.write(f"- Filename: {metadata.get('file_filename', 'N/A')}")
                    st.write(f"- File Type: {metadata.get('file_type', 'N/A')}")
                    st.write(f"- Processed At: {metadata.get('processed_at', 'N/A')}")
                    st.markdown("---")
        else:
            st.info("No results found")
        with st.expander("View raw JSON response"):
            st.json(results)
    st.caption("ClipABit - Powered by CLIP embeddings and semantic search")