"""Video script generation controller."""

from __future__ import annotations

import os
import re

from gtts import gTTS

from .BaseController import BaseController
from ..Stores.LLM import GenAIProvider
from ..models.prompts import VideoScriptTemplate


class VideoScriptController(BaseController):
    def __init__(
        self,
        provider: GenAIProvider,
        lang: str = "en",
        video_topic: str = None,
        style: str = "Simple & Clear",
        duration: int = 3,
    ) -> None:
        super().__init__()

        self.lang = lang if lang != "auto" else "en"
        self.provider = provider
        self.video_topic = video_topic
        self.style = style
        self.duration = duration

        self.template = (
            VideoScriptTemplate.AR.value
            if self.lang.lower().strip() == "ar"
            else VideoScriptTemplate.EN.value
        )

    # ── Word count helpers ────────────────────────────────────────────────────

    def calculate_words(self, words_per_minute: int = 130):
        total_words = self.duration * words_per_minute
        intro_words = int(total_words * 0.2)
        conclusion_words = int(total_words * 0.2)
        main_words = total_words - intro_words - conclusion_words
        return total_words, intro_words, main_words, conclusion_words

    # ── Script generation ─────────────────────────────────────────────────────

    def GenerateScript(self, words_per_minute: int = 130):  # noqa: N802
        total_words, intro_words, main_words, conclusion_words = self.calculate_words(words_per_minute)

        prompt = self.template.format(
            topic=self.video_topic,
            style=self.style,
            total_minutes=self.duration,
            intro_words=intro_words,
            main_words=main_words,
            conclusion_words=conclusion_words,
        )

        response = self.provider.generate_chunks(prompt, temperature=0.5)
        json_output = self.script_to_json(response) if response else {}
        return response, json_output

    # ── Parsing ───────────────────────────────────────────────────────────────

    def script_to_json(self, raw_script: str) -> dict:
        sections: list[dict] = []
        current_section: dict | None = None
        script_text = ""
        script_chunks: list[str] = []

        cleaned_script = re.sub(r"\*\*|\*", "", raw_script)

        for line in cleaned_script.split("\n"):
            line = line.strip()
            if not line or line.startswith("---"):
                continue

            if line.startswith("[INTRO]") or line.startswith("[MAIN]") or line.startswith("[CONCLUSION]"):
                if current_section is not None:
                    sections.append(current_section)
                current_section = {"title": line.strip("[]"), "parts": []}
                continue

            if current_section is None:
                continue

            if line.startswith("NARRATOR:"):
                narrator_text = line.replace("NARRATOR:", "").strip()
                current_section["parts"].append({"type": "narrator", "text": narrator_text})
                script_text += narrator_text + "\n\n"
                script_chunks.append(narrator_text)
            elif line.startswith("VISUALS:"):
                current_section["parts"].append({
                    "type": "visuals",
                    "text": line.replace("VISUALS:", "").strip(),
                })
            elif line.startswith("TEXT:"):
                current_section["parts"].append({
                    "type": "text",
                    "text": line.replace("TEXT:", "").strip(),
                })

        if current_section is not None:
            sections.append(current_section)

        # Extract named sections
        intro_text, body_text, conclusion_text = "", "", ""
        for section in sections:
            narrator_parts = [p["text"] for p in section["parts"] if p["type"] == "narrator"]
            text_block = "\n\n".join(narrator_parts)
            if section["title"] == "INTRO":
                intro_text = text_block
            elif section["title"] == "MAIN":
                body_text = text_block
            elif section["title"] == "CONCLUSION":
                conclusion_text = text_block

        return {
            "title": self.video_topic,
            "style": self.style,
            "duration_minutes": self.duration,
            "sections": {
                "intro": intro_text.strip(),
                "body": body_text.strip(),
                "conclusion": conclusion_text.strip(),
            },
            "narration": script_text.strip(),
            "chunks": script_chunks,
        }

    # ── Audio ─────────────────────────────────────────────────────────────────

    def video_to_audio(self, script_text: str, lang: str = "en") -> str:
        """Convert narration text to an MP3 file and return the path."""
        os.makedirs(self.temp_dir, exist_ok=True)
        random_name = self.generate_random_string(8)
        filename = os.path.join(self.temp_dir, f"{random_name}_video_script.mp3")
        tts = gTTS(text=script_text, lang=lang, tld="com" if lang != "ar" else None)
        tts.save(filename)
        return filename


# Backward-compat alias (old typo name)
VideoSriptGenController = VideoScriptController
