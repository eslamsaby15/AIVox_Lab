import streamlit as st
from src.pages import (
    setup_page,
    summarizer_page,
    Diarizationr_page,
    VideoScriptGenerationPage,
    PodcastSriptPage,
    Translation_page,
    QA_Page,
    SentimentAnalysis_page,
    TopicTagging_page,
    MiniQuiz_page,
)


def main():
    setup_page()

    feature = st.session_state.get("features", "🏡 Home")

    page_map = {
        "🏡 Home": None,               # rendered inside setup_page()
        "📝 Summarize": summarizer_page,
        "🌍 Translation": Translation_page,
        "📊 Sentiment Analysis": SentimentAnalysis_page,
        "🎧 Podcast Generator": PodcastSriptPage,
        "📽️ Video Script Generator": VideoScriptGenerationPage,
        "❓ Interactive Voice Quiz": QA_Page,
        "🔊 Speaker Diarization": Diarizationr_page,
        "🏷️ Topic Tagging": TopicTagging_page,
        "🧩 Multi Quiz": MiniQuiz_page,
    }

    page_fn = page_map.get(feature)
    if page_fn is not None:
        try:
            page_fn()
        except Exception as exc:
            st.error(f"⚠️ An unexpected error occurred: {exc}")
            st.info("Please check your API keys and try again.")


if __name__ == "__main__":
    main()