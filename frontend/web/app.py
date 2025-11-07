"""Small Streamlit frontend for ClipABit.

Usage: run `streamlit run app.py` in this directory. The app lets you upload a video file
and sends it to your Modal backend API.

The backend must accept multipart/form-data with a `file` field for uploads.

This file is intentionally self-contained and uses only `requests` and `streamlit`.
"""

import base64
import io
import requests
import streamlit as st


st.set_page_config(page_title="ClipABit - Video Upload", layout="centered")

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
                        st.success("✅ Upload successful!")
                        try:
                            data = resp.json()

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
                    st.error(f"❌ Upload failed: {e}")


st.markdown("---")
st.markdown(
    """
    **Notes:**
    - The backend must accept a multipart form file field named `file`
    - Make sure your Modal backend is running (`modal serve main.py`)
    - Update the API URL above to match your Modal deployment URL
    """
)
