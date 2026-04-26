"""Translation page — supports multi-language LLM translation and EN→AR classical."""

from __future__ import annotations

import streamlit as st

from ..models.ENUMS.InputENUm import InputTypes
from ..controllers import Youtube, Wav2VecTranscriber
from ..tasks import TranslationTask

_LANGUAGES = [
    "Arabic", "French", "Spanish", "German", "Chinese", "Japanese",
    "Portuguese", "Russian", "Italian", "Turkish", "Hindi", "Korean",
]


def Translation_page() -> None:  # noqa: N802
    st.subheader("🌍 Translation")

    # ── Session state defaults ─────────────────────────────────────────────
    for key in ("transcript", "translation"):
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Input source ───────────────────────────────────────────────────────
    input_type = st.selectbox("Input type", ["Upload file", "YouTube link"], key="trans_input_type")
    youtube_link = file_uploaded = None

    if input_type == "Upload file":
        file_uploaded = st.file_uploader(
            "📂 Upload audio/video",
            type=[InputTypes.WAV.value, InputTypes.MKV.value,
                  InputTypes.MP4.value, InputTypes.MP3.value],
            key="trans_uploader",
        )
    else:
        youtube_link = st.text_input("🔗 Paste YouTube link", key="trans_yt_link")

    # ── Translation settings ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("⚙️ Mode", ["llm", "classic (EN→AR only)"], index=0, key="trans_mode")
    with col2:
        target_language = st.selectbox(
            "🌐 Target Language",
            _LANGUAGES,
            disabled=mode.startswith("classic"),
            key="trans_lang",
        )

    # ── Process ────────────────────────────────────────────────────────────
    if st.button("🚀 Translate", key="trans_btn"):
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
                st.session_state.transcript = transcriber.transcribe(wav_file)
            st.success("✅ Transcription complete!")
        except Exception as exc:
            st.error(f"❌ Transcription failed: {exc}")
            return

        try:
            with st.spinner(f"🌍 Translating to {target_language}..."):
                actual_mode = "classic" if mode.startswith("classic") else "llm"
                task = TranslationTask(
                    mode=actual_mode,
                    provider_name=st.session_state.get("generation_provider", "Gemini"),
                    target_language=target_language,
                )
                st.session_state.translation = task.run(st.session_state.transcript)
            st.success("✅ Translation complete!")
        except Exception as exc:
            st.error(f"❌ Translation failed: {exc}")
            return

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.transcript:
        with st.expander("📜 Transcript", expanded=False):
            st.text_area("", st.session_state.transcript, height=180, key="trans_transcript_view")

    if st.session_state.translation:
        st.subheader("🌍 Translation")
        st.text_area("", st.session_state.translation, height=220, key="trans_result_view")

        st.download_button(
            "📥 Download Translation",
            st.session_state.translation,
            file_name="translation.txt",
            mime="text/plain",
        )
