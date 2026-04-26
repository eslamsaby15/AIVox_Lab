"""Summarizer page."""

from __future__ import annotations

import streamlit as st

from ..models.ENUMS.InputENUm import InputTypes
from ..controllers import Youtube, Wav2VecTranscriber
from ..tasks import SummarizerTask


def summarizer_page() -> None:
    st.subheader("📝 Summarize")

    for key in ("sum_transcript", "sum_summary"):
        if key not in st.session_state:
            st.session_state[key] = None

    input_type = st.selectbox("Input type", ["Upload file", "YouTube link"], key="sum_input_type")
    youtube_link = file_uploaded = None

    if input_type == "Upload file":
        file_uploaded = st.file_uploader(
            "📂 Upload audio/video",
            type=[InputTypes.WAV.value, InputTypes.MKV.value,
                  InputTypes.MP4.value, InputTypes.MP3.value],
            key="sum_uploader",
        )
    else:
        youtube_link = st.text_input("🔗 Paste YouTube link", key="sum_yt_link")

    col1, col2 = st.columns(2)
    with col1:
        lang_choice = st.selectbox("🌐 Language", ["auto", "en", "ar"], key="sum_lang")
    with col2:
        mode = st.radio("⚙️ Mode", ["llm", "classic"], index=0, key="sum_mode")

    if st.button("🚀 Summarize", key="sum_btn"):
        if not youtube_link and not file_uploaded:
            st.warning("⚠️ Please provide a file or YouTube link.")
            return

        try:
            yt = Youtube()
            with st.spinner("📥 Preparing audio..."):
                wav_file = yt.Download(youtube_link) if youtube_link else yt.save_dir(file_uploaded)
        except Exception as exc:
            st.error(f"❌ Failed to load audio: {exc}")
            return

        try:
            with st.spinner("📝 Transcribing..."):
                transcriber = Wav2VecTranscriber()
                st.session_state.sum_transcript = transcriber.transcribe(wav_file)
            st.success("✅ Transcription complete!")
        except Exception as exc:
            st.error(f"❌ Transcription failed: {exc}")
            return

        try:
            with st.spinner("📌 Summarizing..."):
                task = SummarizerTask(
                    lang=lang_choice,
                    mode=mode,
                    provider_name=st.session_state.get("generation_provider", "Gemini"),
                )
                st.session_state.sum_summary = task.run(st.session_state.sum_transcript)
            st.success("✅ Summary generated!")
        except Exception as exc:
            st.error(f"❌ Summarization failed: {exc}")
            return

    if st.session_state.sum_transcript:
        with st.expander("📜 Transcript", expanded=False):
            st.text_area("", st.session_state.sum_transcript, height=180, key="sum_transcript_view")

    if st.session_state.sum_summary:
        st.subheader("📌 Summary")
        st.text_area("", st.session_state.sum_summary, height=220, key="sum_result_view")
        st.download_button(
            "📥 Download Summary",
            st.session_state.sum_summary,
            file_name="summary.txt",
            mime="text/plain",
        )
