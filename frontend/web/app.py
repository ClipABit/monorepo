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


def search_videos(query: str):
    """Send search query to backend."""
    try:
        resp = requests.post(SEARCH_API_URL, json={"query": query}, timeout=30)
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
                with st.spinner("Uploading..."):
                    try:
                        resp = upload_file_to_backend(uploaded_bytes, uploaded.name, uploaded.type)
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("status") == "processing":
                                job_id = data.get("job_id")
                                st.toast(f"✓ Video uploaded! Job ID: {job_id}", icon="✅")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Upload failed")
                        else:
                            st.error(f"Upload failed with status {resp.status_code}")
                    except requests.RequestException as e:
                        st.error(f"Upload failed: {e}")
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


# Main UI
st.title("🎬 ClipABit")
st.subheader("Semantic Video Search")

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
        # Display query echo
        if "query" in results:
            st.info(f"Query: {results['query']}")
        
        # Display full JSON response
        with st.expander("View raw JSON response", expanded=True):
            st.json(results)

# Footer
st.markdown("---")
st.caption("ClipABit - Powered by CLIP embeddings and semantic search")

