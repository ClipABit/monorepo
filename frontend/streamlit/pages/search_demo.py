import io
import time
import requests
import streamlit as st
from config import Config

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# API endpoints from config
SEARCH_API_URL = Config.SEARCH_API_URL
UPLOAD_API_URL = Config.UPLOAD_API_URL
STATUS_API_URL = Config.STATUS_API_URL
LIST_VIDEOS_API_URL = Config.LIST_VIDEOS_API_URL
NAMESPACE = Config.NAMESPACE
ENVIRONMENT = Config.ENVIRONMENT


def search_videos(query: str):
    """Send search query to backend."""
    try:
        resp = requests.get(SEARCH_API_URL, params={"query": query, "namespace": NAMESPACE}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"Search failed with status {resp.status_code}"}
    except requests.RequestException as e:
        return {"error": str(e)}


@st.cache_data(ttl=60, show_spinner="Fetching all videos in repository...")
def fetch_all_videos():
    """Fetch all videos from the backend."""
    try:
        resp = requests.get(LIST_VIDEOS_API_URL, params={"namespace": NAMESPACE}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("videos", [])
        return []
    except requests.RequestException:
        return []


def upload_file_to_backend(file_bytes: bytes, filename: str, content_type: str | None = None):
    """Upload file to backend via multipart form-data."""
    files = {"file": (filename, io.BytesIO(file_bytes), content_type or "application/octet-stream")}
    data = {"namespace": NAMESPACE}
    resp = requests.post(UPLOAD_API_URL, files=files, data=data, timeout=300)
    return resp


# Upload dialog
@st.fragment
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
            if st.button("Upload", type="primary", width="stretch"):
                with st.spinner("Uploading..."):
                    try:
                        resp = upload_file_to_backend(uploaded_bytes, uploaded.name, uploaded.type)
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("status") == "processing":
                                job_id = data.get("job_id")
                                st.toast(f"Video uploaded! Job ID: {job_id}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Upload failed")
                        else:
                            st.error(f"Upload failed with status {resp.status_code}. Message: {resp.text}")
                    except requests.RequestException as e:
                        st.error(f"Upload failed: {e}")
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


# Main UI
st.title("ClipABit")
st.subheader("Semantic Video Search - Demo")

# Upload button row
up_col1, up_col2, up_col3 = st.columns([1, 1, 6])
# with up_col1:
#     if st.button("Upload", disabled=True, width="stretch"):
#         upload_dialog()

upload_enabled = (ENVIRONMENT != "prod")
if upload_enabled:
    with up_col1:
        # upload disabled in prod env
        if st.button("Upload", disabled=(False), width="stretch"):
            upload_dialog()
else:
    st.text("The repository below mimics the footage you would have in video editor's media pool. "
            "Try searching for specific actions, settings, objects in the videos using natural language! "
            "We'd appreciate any feedback you may have.")
# with up_col2:
#     if st.button("Feedback", width="stretch"):
#         st.switch_page("pages/feedback.py")
        
# insert vertical spaces
st.write("")
st.write("")

# Header with search and clear button
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    search_query = st.text_input(
        "Search",
        placeholder="Type to search...",
        label_visibility="collapsed"
    )

with col2:
    if st.button("Search", type="primary", use_container_width=True):
        if search_query:
            with st.spinner("Searching..."):
                results = search_videos(search_query)
                st.session_state.search_results = results
        else:
            st.warning("Please enter a search query")

with col3:
    if st.button("Clear", use_container_width=True):
        st.session_state.search_results = None
        st.rerun()


st.markdown("---")

# Custom CSS to force video containers to have a consistent aspect ratio
st.markdown("""
    <style>
    .stVideo {
        aspect-ratio: 16 / 9;
        background-color: #000;
    }
    </style>
""", unsafe_allow_html=True)

# Display results or repository
if st.session_state.search_results:
    st.subheader(f"Search Results for: '{search_query}'")
    
    results_data = st.session_state.search_results
    
    if "error" in results_data:
        st.error(f"Error: {results_data['error']}")
    elif "results" in results_data:
        results = results_data["results"]
        if results:
            cols = st.columns(3)
            for idx, result in enumerate(results):
                metadata = result.get("metadata", {})
                presigned_url = metadata.get("presigned_url")
                start_time = metadata.get("start_time_s", 0)
                filename = metadata.get("file_filename", "Unknown Video")
                score = result.get("score", 0)
                
                if presigned_url:
                    with cols[idx % 3]:
                        with st.expander("Info"):
                            st.write(f"**File:** {filename}")
                            st.write(f"**Score:** {score:.2f}")
                        st.video(presigned_url, start_time=int(start_time))
        else:
            st.info("No matching videos found.")

else:
    st.subheader("Video Repository")
    
    # Fetch and display videos
    videos = fetch_all_videos()

    
    if videos:
        # Create a grid of videos
        cols = st.columns(3)
        for idx, video in enumerate(videos):
            with cols[idx % 3]:
                with st.expander("Info"):
                    st.write(f"**File:** {video['file_name']}")
                st.video(video['presigned_url'])
    else:
        st.info("No videos found in the repository.")

# Footer
st.markdown("---")
st.caption("ClipABit - Powered by CLIP embeddings and semantic search")