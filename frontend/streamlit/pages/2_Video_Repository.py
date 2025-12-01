import streamlit as st
import requests

# API endpoints
LIST_VIDEOS_API_URL = "https://clipabit01--clipabit-server-list-videos-dev.modal.run"

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

st.subheader("Video Repository")
    
# Fetch and display videos
videos = fetch_all_videos()

if videos:
    # Create a grid of videos
    cols = st.columns(3)
    for idx, video in enumerate(videos):
        with cols[idx % 3]:
            st.caption(f"**{video['file_name']}**")
            st.video(video['presigned_url'])
else:
    st.info("No videos found in the repository.")

# Footer
st.markdown("---")
st.caption("ClipABit - Powered by CLIP embeddings and semantic search")