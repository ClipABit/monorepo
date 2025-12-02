import streamlit as st

st.set_page_config(
    page_title="ClipABit",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with st.sidebar:
    st.success("☝️ Explore the demo and features")

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Make the main content a bit wider and cleaner */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        /* Cards */
        .clip-card {
            border-radius: 18px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(250, 250, 250, 0.12);
            background: rgba(250, 250, 250, 0.03);
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            margin-top: 1rem;
            margin-bottom: 0.6rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #9CA3AF;
            max-width: 32rem;
        }

        .hero-tagline {
            font-size: 0.9rem;
            color: #A0A1A0;
            margin-top: 0.4rem;
            margin-bottom: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown('<div class="hero-title">Find the perfect clip in seconds.</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-subtitle">
        ClipABit is like <b>Ctrl + F for your footage</b>.  
        Search across hours of raw video using natural language and jump straight to the moments that matter.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-tagline">
        No more scrubbing timelines. No more guessing. Just type what you remember and we’ll find it.
    </div>
    """,
    unsafe_allow_html=True,
)


if st.button("🚀 Try the demo", width="stretch"):
        st.switch_page("pages/1_Search_Demo.py")  # adjust or remove if you don't use multipage

st.markdown("---")

# Section: What is ClipABit?
st.subheader("What is ClipABit?")
st.markdown(
    """
    <div class="clip-card">
    Ever watched a long YouTube video and thought about how hard editors had to work to find the right clips?

    **ClipABit** makes that process radically faster by letting editors:
    - 🔍 **Search for whatever** - like using Google!
    - 🧠 **Identify faces and people** thanks to our facial recognition algorithms
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Section: Demo overview
st.subheader("What’s in this demo?")

demo_col1, demo_col2 = st.columns(2)

with demo_col1:
    st.markdown(
        """
        <div class="clip-card">
          <h3>🎯 Curated sample videos</h3>
          <p>We've included a set of short, diverse clips designed to show:</p>
          <ul>
            <li>Different scenes & subjects</li>
            <li>Lighting and motion variety</li>
            <li>How accurate semantic search is</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with demo_col2:
    st.markdown(
        """
        <div class="clip-card">
          <h3>🔎 Test search quality</h3>
          <p>Try prompts like:</p>
          <ul>
            <li>“host pointing at screen”</li>
            <li>“wide shot of skyline”</li>
            <li>“cutaway of audience reacting”</li>
          </ul>
          <p>The goal: to showcase the speed & accuracy of our semantic search.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Section: What we're trying to do
problem_col1, problem_col2 = st.columns(2)

with problem_col1:
    st.subheader("What problem are we solving?")
    st.markdown(
        """
        <div class="clip-card">
        Editing takes so long not just because of creative decisions, but because of searching:

        - Hours of raw recordings are dumped into a bin  
        - Editors need to manually scrub, tag, and bookmark points of interest
        - Finding a single moment can mean replaying the same footage over and over  

        We want to remove that friction so creators can go from recording to editing as fast as possible.
        </div>
        """,
        unsafe_allow_html=True,
    )

with problem_col2:
    st.subheader("What’s next for ClipABit?")
    st.markdown(
        """
        <div class="clip-card">
          <ul>
            <li>🧩 Integrations with DaVinci Resolve, Adobe Premiere Pro, and Final Cut Pro</li>
            <li>⚙️ Multithreading & batching to handle huge libraries faster</li>
            <li>🧬 Smarter models to combine audio and visual understanding</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.caption("ClipABit - Powered by CLIP embeddings and semantic search")
