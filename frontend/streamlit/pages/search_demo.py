import io
import time
import requests
import streamlit as st
from config import Config

REPO_PAGE_SIZE = 20

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'repo_videos' not in st.session_state:
    st.session_state.repo_videos = []
if 'repo_next_token' not in st.session_state:
    st.session_state.repo_next_token = None
if 'repo_initialized' not in st.session_state:
    st.session_state.repo_initialized = False
if 'repo_page_size' not in st.session_state:
    st.session_state.repo_page_size = REPO_PAGE_SIZE

# Configs
SEARCH_API_URL = Config.SEARCH_API_URL
UPLOAD_API_URL = Config.UPLOAD_API_URL
STATUS_API_URL = Config.STATUS_API_URL
LIST_VIDEOS_API_URL = Config.LIST_VIDEOS_API_URL
DELETE_VIDEO_API_URL = Config.DELETE_VIDEO_API_URL
NAMESPACE = Config.NAMESPACE
ENVIRONMENT = Config.ENVIRONMENT
IS_INTERNAL_ENV = Config.IS_INTERNAL_ENV


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

def fetch_videos_page(page_token: str | None = None, page_size: int = REPO_PAGE_SIZE):
    """Fetch a single page of videos from the backend."""
    params = {
        "namespace": NAMESPACE,
        "page_size": page_size,
    }
    if page_token:
        params["page_token"] = page_token

    try:
        resp = requests.get(LIST_VIDEOS_API_URL, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "videos": data.get("videos", []),
                "next_page_token": data.get("next_page_token"),
            }
        return {"error": f"Fetch failed with status {resp.status_code}"}
    except requests.RequestException as e:
        return {"error": str(e)}


def load_repository_page(page_token: str | None = None, append: bool = False) -> tuple[bool, str | None]:
    """Load a page of videos into session state."""
    page_size = st.session_state.repo_page_size
    result = fetch_videos_page(page_token=page_token, page_size=page_size)

    if "error" in result:
        return False, result["error"]

    videos = result.get("videos", [])
    next_token = result.get("next_page_token")

    if append:
        st.session_state.repo_videos.extend(videos)
    else:
        st.session_state.repo_videos = videos

    st.session_state.repo_next_token = next_token
    st.session_state.repo_initialized = True
    return True, None


def reset_repository_state() -> None:
    """Reset cached repository data so it reloads on next view."""
    st.session_state.repo_videos = []
    st.session_state.repo_next_token = None
    st.session_state.repo_initialized = False


def upload_files_to_backend(files_data: list[tuple[bytes, str, str]]):
    """Upload single or multiple files to backend via multipart form-data."""
    files = [
        ("files", (filename, io.BytesIO(file_bytes), content_type or "application/octet-stream"))
        for file_bytes, filename, content_type in files_data
    ]
    data = {"namespace": NAMESPACE}

    # Dynamic timeout: 300s base + 30s per file, minimum 600s
    # Handles large batches: 50 files = 1800s (30min), 200 files = 6300s (105min)
    timeout = max(600, 300 + (len(files) * 30))

    resp = requests.post(UPLOAD_API_URL, files=files, data=data, timeout=timeout)
    return resp


def poll_job_status(job_id: str, max_wait: int = 300):
    """Poll job status until completion or timeout."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(STATUS_API_URL, params={"job_id": job_id}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                if status in ["completed", "partial", "failed"]:
                    return data
            else:
                return {"error": f"Status check failed with status {resp.status_code}"}
        except requests.RequestException as e:
            return {"error": str(e)}
        time.sleep(2)
    return {"error": "Timeout waiting for completion"}


def delete_video(hashed_identifier: str, filename: str):
    """Delete video via API call."""

    if not IS_INTERNAL_ENV:
        st.toast(f"Deletion not allowed in {ENVIRONMENT} environment", icon="🚫")
        return

    try:
        resp = requests.delete(
            DELETE_VIDEO_API_URL,
            params={
                "hashed_identifier": hashed_identifier,
                "filename": filename,
                "namespace": NAMESPACE
            },
            timeout=30
        )
        if resp.status_code == 200:
            _ = resp.json() # TODO: should do smth with result
            st.toast(f"✅ Video '{filename}' deleted successfully!", icon="✅")
            st.session_state.search_results = None  # Clear search results to refresh the display
            reset_repository_state()
            st.rerun()  # Force refresh UI
        elif resp.status_code == 404:
            st.toast(f"⚠️ Video '{filename}' not found", icon="⚠️")
        elif resp.status_code == 403:
            st.toast(f"🚫 Deletion not allowed in {ENVIRONMENT} environment", icon="🚫")
        else:
            st.toast(f"❌ Delete failed with status {resp.status_code}", icon="❌")
    except requests.RequestException as e:
        st.toast(f"❌ Network error: {str(e)}", icon="❌")

# Upload dialog (handles single and multiple files)
@st.fragment
@st.dialog("Upload Videos")
def upload_dialog():
    st.write("Upload one or more videos to add them to the searchable database.")

    uploaded_files = st.file_uploader(
        "Choose video file(s)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"**Selected Files:** {len(uploaded_files)}")

        # Collect file metadata without loading all bytes into memory
        total_size = 0
        file_info = []
        for file in uploaded_files:
            file_size = file.size
            total_size += file_size
            file_info.append({
                "name": file.name,
                "file_obj": file,  # Store reference, not bytes
                "type": file.type,
                "size": file_size
            })

        st.write(f"**Total Size:** {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")

        with st.expander("File Details"):
            for info in file_info:
                st.write(f"- {info['name']} ({info['size'] / 1024 / 1024:.2f} MB)")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Upload", type="primary", use_container_width=True):
                with st.spinner(f"Uploading {len(file_info)} video(s)..."):
                    try:
                        # Read bytes only when uploading (lazy loading)
                        files_data = [(info["file_obj"].read(), info["name"], info["type"]) for info in file_info]
                        resp = upload_files_to_backend(files_data)

                        if resp.status_code == 200:
                            data = resp.json()

                            # Check if single video or batch
                            if "job_id" in data:
                                # Single video
                                job_id = data.get("job_id")
                                st.success(f"Video uploaded! Job ID: {job_id}")
                                reset_repository_state()
                                time.sleep(2)
                                st.rerun()
                            elif "batch_job_id" in data:
                                # Batch upload
                                batch_job_id = data.get("batch_job_id")
                                st.success(f"Batch uploaded! Job ID: {batch_job_id}")
                                st.write(f"Processing {data.get('total_videos', 0)} videos...")

                                progress_bar = st.progress(0)
                                status_text = st.empty()

                                while True:
                                    status_data = poll_job_status(batch_job_id, max_wait=5)

                                    if "error" in status_data:
                                        st.error(f"Error checking status: {status_data['error']}")
                                        break

                                    status = status_data.get("status")
                                    progress = status_data.get("progress_percent", 0)
                                    completed = status_data.get("completed_count", 0)
                                    failed = status_data.get("failed_count", 0)
                                    processing = status_data.get("processing_count", 0)

                                    progress_bar.progress(progress / 100.0)
                                    status_text.text(
                                        f"Status: {status} | "
                                        f"Completed: {completed} | "
                                        f"Failed: {failed} | "
                                        f"Processing: {processing}"
                                    )

                                    if status in ["completed", "partial", "failed"]:
                                        if status == "completed":
                                            st.success(f"All {completed} videos processed successfully!")
                                            metrics = status_data.get("metrics", {})
                                            st.write(f"Total chunks: {metrics.get('total_chunks', 0)}")
                                            st.write(f"Total frames: {metrics.get('total_frames', 0)}")
                                        elif status == "partial":
                                            st.warning(
                                                f"Batch completed with {completed} successes and {failed} failures"
                                            )
                                            failed_jobs = status_data.get("failed_jobs", [])
                                            if failed_jobs:
                                                with st.expander("Failed Videos"):
                                                    for job in failed_jobs:
                                                        st.write(f"- {job.get('filename')}: {job.get('error')}")
                                        else:
                                            st.error(f"All {failed} videos failed to process")
                                            failed_jobs = status_data.get("failed_jobs", [])
                                            if failed_jobs:
                                                with st.expander("Failed Videos"):
                                                    for job in failed_jobs:
                                                        st.write(f"- {job.get('filename')}: {job.get('error')}")

                                        reset_repository_state()
                                        time.sleep(2)
                                        st.rerun()
                                        break

                                    time.sleep(2)
                        else:
                            st.error(f"Upload failed with status {resp.status_code}. Message: {resp.text}")
                    except requests.RequestException as e:
                        st.error(f"Upload failed: {e}")

        with col2:
            if st.button("Cancel", use_container_width=True):
                st.rerun()

# Delete confirmation dialog
@st.dialog("Delete Video")
def delete_confirmation_dialog(hashed_identifier: str, filename: str):
    """Show delete confirmation dialog."""
    st.write(f"Are you sure you want to delete **{filename}**?")
    st.warning("⚠️ This action cannot be undone!")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Delete", type="primary", use_container_width=True):
            delete_video(hashed_identifier, filename)


# Main UI
st.title("ClipABit")
st.subheader("Semantic Video Search - Demo")

# Upload button row
up_col1, up_col2, up_col3 = st.columns([1, 1, 6])

# upload button in internal envs else info text
if IS_INTERNAL_ENV:
    with up_col1:
        if st.button("Upload", disabled=(False), use_container_width=True):
            upload_dialog()
else:
    st.text("The repository below mimics the footage you would have in your video editor's media pool. "
            "Try searching for specific actions, settings, objects in the videos using natural language! "
            "We'd appreciate any feedback you may have.")

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
                with cols[idx%3]:
                    metadata = result.get("metadata", {})
                    presigned_url = metadata.get("presigned_url")
                    start_time = metadata.get("start_time_s", 0)
                    filename = metadata.get("file_filename", "Unknown Video")
                    hashed_identifier = metadata.get("hashed_identifier", "")
                    score = result.get("score", 0)

                    # Video info and delete button row
                    if IS_INTERNAL_ENV:
                        info_col, delete_col = st.columns([3, 1]) # NOTE: reenable with delete

                        with info_col:
                            with st.expander("Info"):
                                st.write(f"**File:** {filename}")
                                st.write(f"Score: {score:.2f}")
                        with delete_col:
                            if hashed_identifier:
                                if st.button("🗑️", key=f"delete_search_{idx}", help=f"Delete {filename}"):
                                    delete_confirmation_dialog(hashed_identifier, filename)
                    else:
                        with st.expander("Info"):
                            st.write(f"**File:** {filename}")
                            st.write(f"**Score:** {score:.2f}")
                    st.video(presigned_url, start_time=int(start_time))

        else:
            st.info("No matching videos found.")

else:
    st.subheader("Video Repository")

    if not st.session_state.repo_initialized:
        with st.spinner("Loading videos..."):
            success, error = load_repository_page()
        if not success and error:
            st.error(f"Failed to load videos: {error}")

    videos = st.session_state.repo_videos

    if videos:
        # Create a grid of videos
        cols = st.columns(3)
        for idx, video in enumerate(videos):
            with cols[idx%3]:
                # Video info and delete button row
                if IS_INTERNAL_ENV:
                    info_col, delete_col = st.columns([3, 1])
                    with info_col:
                        with st.expander("Info"):
                            st.write(f"**File:** {video['file_name']}")
                    with delete_col:
                        if video.get('hashed_identifier'):
                            if st.button("🗑️", key=f"delete_repo_{idx}", help=f"Delete {video['file_name']}"):
                                delete_confirmation_dialog(video['hashed_identifier'], video['file_name'])
                else:
                    with st.expander("Info"):
                        st.write(f"**File:** {video['file_name']}")

                st.video(video['presigned_url'])
    else:
        st.info("No videos found in the repository.")

    if st.session_state.repo_next_token:
        if st.button("Load more videos", use_container_width=True):
            with st.spinner("Loading more videos..."):
                success, error = load_repository_page(
                    page_token=st.session_state.repo_next_token,
                    append=True
                )
            if not success and error:
                st.error(f"Failed to load additional videos: {error}")
            else:
                st.rerun()

# Footer
st.markdown("---")
st.caption("ClipABit - Powered by CLIP embeddings and semantic search")
