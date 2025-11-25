import io
import time
import requests
import streamlit as st


st.set_page_config(page_title="ClipABit - Video Upload", layout="centered")

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

st.title("ClipABit — Video Uploader")
st.caption("Upload a video file and send it to your Modal backend for processing.")

api_url = st.text_input(
    "Backend API URL",
    value="https://clipabit01--clipabit-server-upload-dev.modal.run",
    help="Full URL of the Modal backend endpoint that accepts multipart file uploads.",
)


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

                # Job completed or failed
                if status in ["completed", "failed"]:
                    return data

            time.sleep(poll_interval)
        except requests.RequestException:
            time.sleep(poll_interval)

    # Timeout
    return {"job_id": job_id, "status": "timeout", "message": "Job polling timed out"}


# API endpoint for search
SEARCH_API_URL = "https://clipabit01--clipabit-server-search-dev.modal.run"


def search_videos(query: str):
    """Send search query to backend."""
    try:
        resp = requests.post(SEARCH_API_URL, params={"query": query}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"Search failed with status {resp.status_code}"}
    except requests.RequestException as e:
        return {"error": str(e)}


# Upload dialog
@st.dialog("Upload Video")
def upload_dialog():
    """Dialog for uploading videos."""
    st.write("Upload a video to add it to the searchable database.")

    dialog_uploaded = st.file_uploader(
        "Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"], key="dialog_uploader"
    )

    if dialog_uploaded is not None:
        dialog_bytes = dialog_uploaded.read()

        # Show video preview
        try:
            st.video(io.BytesIO(dialog_bytes))
        except Exception:
            st.info("Preview not available for this format.")

        # File metadata
        st.write(f"**Filename:** {dialog_uploaded.name}")
        st.write(f"**Size:** {len(dialog_bytes):,} bytes ({len(dialog_bytes) / 1024 / 1024:.2f} MB)")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Upload", type="primary", use_container_width=True):
                with st.spinner("Uploading..."):
                    try:
                        resp = upload_file_to_backend(api_url, dialog_bytes, dialog_uploaded.name, dialog_uploaded.type)

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


# File uploader
uploaded = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"])

if uploaded is not None:
    # Read bytes once so we can both preview and upload
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
    if st.button("Upload to Backend", type="primary"):
        if not api_url:
            st.error("Please set the backend API URL first.")
        else:
            with st.spinner("Uploading video to backend..."):
                try:
                    resp = upload_file_to_backend(api_url, uploaded_bytes, uploaded.name, uploaded.type)

                    # Handle response
                    if resp.status_code == 200:
                        try:
                            upload_data = resp.json()

                            # Check if job was spawned
                            if upload_data.get("status") == "processing":
                                job_id = upload_data.get("job_id")
                                st.info(f"Video uploaded successfully. Job ID: {job_id}")

                                # Derive status URL from upload URL
                                status_url = api_url.replace("-upload-", "-status-")

                                # Poll for results
                                with st.spinner("Processing video... This may take a minute."):
                                    data = poll_job_status(status_url, job_id, max_wait=120, poll_interval=2)
                            else:
                                # Legacy: direct response (if not using spawn)
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

                                # Show job ID
                                st.code(f"Job ID: {data.get('job_id', 'N/A')}", language=None)

                                # Show raw JSON in expander
                                with st.expander("View raw response"):
                                    st.json(data)

                                # Display chunk details if available
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


st.markdown("---")
st.markdown(
    """
    **Notes:**
    - The backend must accept a multipart form file field named `file`
    - Make sure your Modal backend is running (`modal serve main.py`)
    - Update the API URL above to match your Modal deployment URL
    """
)
