import streamlit as st
import streamlit.components.v1 as components

about_page = st.Page("pages/about.py", title="About ClipABit", icon="🏠")
demo_page = st.Page("pages/search_demo.py", title="Search Demo", icon="🔎")
feedback_page = st.Page("pages/feedback.py", title="Feedback", icon="💬")

pg = st.navigation([about_page, demo_page, feedback_page])

st.set_page_config(
    page_title="ClipABit",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
pg.run()