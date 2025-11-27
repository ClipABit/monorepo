import io
import time
import requests
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="ClipABit",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# API endpoints
SEARCH_API_URL = "https://clipabit01--clipabit-server-search-dev.modal.run"
UPLOAD_API_URL = "https://clipabit01--clipabit-server-upload-dev.modal.run"
STATUS_API_URL = "https://clipabit01--clipabit-server-status-dev.modal.run"
LIST_VIDEOS_API_URL = "https://clipabit01--clipabit-server-list-videos-dev.modal.run"


def search_videos(query: str):
    """Send search query to backend."""
    try:
        resp = requests.get(SEARCH_API_URL, params={"query": query}, timeout=30)
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
        resp = requests.get(LIST_VIDEOS_API_URL, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("videos", [])
        return []
    except requests.RequestException:
        return []


def upload_file_to_backend(file_bytes: bytes, filename: str, content_type: str | None = None):
    """Upload file to backend via multipart form-data."""
    files = {"file": (filename, io.BytesIO(file_bytes), content_type or "application/octet-stream")}
    resp = requests.post(UPLOAD_API_URL, files=files, timeout=300)
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
            if st.button("Upload", type="primary", use_container_width=True):
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
                            st.error(f"Upload failed with status {resp.status_code}")
                    except requests.RequestException as e:
                        st.error(f"Upload failed: {e}")
        
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()


@st.dialog("Search Results", width="large")
def search_results_dialog(query: str):
    st.write(f"Results for: **{query}**")
    
    with st.spinner("Searching..."):
        results = search_videos(query)
    
    if "error" in results:
        st.error(f"Error: {results['error']}")
    else:
        # Display full JSON response for now as requested
        # "Use the video name and hashed id wherever you need metadata/ids about the video in the frontend"
        # Assuming search results will eventually return similar structures
        st.json(results)


# Main UI
st.title("ClipABit")
st.subheader("Semantic Video Search - Demo")

# Header with search and upload button
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    search_query = st.text_input(
        "Search",
        placeholder="Search for video content...",
        label_visibility="collapsed"
    )

with col2:
    if st.button("Search", type="primary", use_container_width=True):
        if search_query:
            search_results_dialog(search_query)
        else:
            st.warning("Please enter a search query")

with col3:
    if st.button("Upload", use_container_width=True):
        upload_dialog()


st.markdown("---")
st.subheader("Video Repository")

# Fetch and display videos
videos = fetch_all_videos()

# Custom CSS to force video containers to have a consistent aspect ratio
st.markdown("""
    <style>
    .stVideo {
        aspect-ratio: 16 / 9;
        background-color: #000;
    }
    </style>
""", unsafe_allow_html=True)

if videos:
    # Create a grid of videos
    cols = st.columns(3)
    for idx, video in enumerate(videos):
        with cols[idx % 3]:
            st.video(video['presigned_url'])
else:
    st.info("No videos found in the repository.")

# Footer
st.markdown("---")
st.caption("ClipABit - Powered by CLIP embeddings and semantic search")