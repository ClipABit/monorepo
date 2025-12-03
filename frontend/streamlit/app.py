import streamlit as st

demo_page = st.Page("pages/search_demo.py", title="Search Demo", icon="🔎")
about_page = st.Page("pages/about.py", title="About ClipABit", icon="ℹ️")
pg = st.navigation([about_page, demo_page])

st.set_page_config(
    page_title="ClipABit",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
pg.run()