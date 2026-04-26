"""
App setup: page config, sidebar navigation, and home screen.
"""

from __future__ import annotations

import streamlit as st

from ..helpers.config import get_settings, validate_provider_keys


_SIDEBAR_TASKS = [
    {"icon": "🏡", "name": "Home"},
    {"icon": "📝", "name": "Summarize"},
    {"icon": "🌍", "name": "Translation"},
    {"icon": "📊", "name": "Sentiment Analysis"},
    {"icon": "🎧", "name": "Podcast Generator"},
    {"icon": "📽️", "name": "Video Script Generator"},
    {"icon": "❓", "name": "Interactive Voice Quiz"},
    {"icon": "🔊", "name": "Speaker Diarization"},
    {"icon": "🏷️", "name": "Topic Tagging"},
    {"icon": "🧩", "name": "Multi Quiz"},
]


def _init_session_state() -> None:
    defaults = {
        "features": "🏡 Home",
        "generation_provider": "Gemini",
        "embedding_provider": "Cohere",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sidebar() -> None:
    cfg = get_settings()

    feature_names = [f"{t['icon']} {t['name']}" for t in _SIDEBAR_TASKS]
    current = st.session_state["features"]
    if current not in feature_names:
        current = feature_names[0]

    selected = st.sidebar.selectbox(
        "🗂️ Select Task",
        feature_names,
        index=feature_names.index(current),
    )
    st.session_state["features"] = selected

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚙️ Provider Settings")

    gen_providers = ["Gemini", "OpenAI", "Cohere"]
    st.session_state["generation_provider"] = st.sidebar.selectbox(
        "Generation Provider",
        gen_providers,
        index=gen_providers.index(st.session_state["generation_provider"]),
    )

    embed_providers = ["Cohere", "Gemini", "OpenAI", "sentence-transformers/all-MiniLM-L6-v2"]
    st.session_state["embedding_provider"] = st.sidebar.selectbox(
        "Embedding Provider",
        embed_providers,
        index=embed_providers.index(st.session_state["embedding_provider"]),
    )

    # Show a warning if the selected provider's API key is missing
    gen_provider = st.session_state["generation_provider"]
    ok, msg = validate_provider_keys(gen_provider)
    if not ok:
        st.sidebar.warning(msg)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**{cfg.APP_NAME}** v{cfg.APP_VERSION}\n\n"
        "Built by **Eslam Sabry**\n\n"
        "🔗 [LinkedIn](https://www.linkedin.com/in/eslamsabryai)  "
        "🔗 [Kaggle](https://www.kaggle.com/eslamsabryelsisi)"
    )


def _render_home() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>🎙️ AIVox Lab</h1>"
        "<p style='text-align:center;color:#888;'>AI-Powered Voice & Text Processing</p>",
        unsafe_allow_html=True,
    )

    try:
        st.image("src/assets/images/b1.png", use_container_width=True)
    except Exception:
        pass  # Image not critical

    st.markdown("### Select a feature to get started:")

    cols = st.columns(3)
    for i, task in enumerate(_SIDEBAR_TASKS):
        label = f"{task['icon']} {task['name']}"
        with cols[i % 3]:
            if st.button(label, key=f"home_{task['name']}", use_container_width=True):
                st.session_state["features"] = label
                st.rerun()


def setup_page() -> None:
    """Configure the Streamlit page and render the sidebar."""
    st.set_page_config(
        page_title="AIVox Lab",
        layout="wide",
        page_icon="🎙️",
        menu_items={"About": "# AIVox Lab\nAI-Powered Voice & Text Processing Platform"},
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 1100px; }
        .stButton > button { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _init_session_state()
    _render_sidebar()

    if st.session_state["features"] == "🏡 Home":
        _render_home()
