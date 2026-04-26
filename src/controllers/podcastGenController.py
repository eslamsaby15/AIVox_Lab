"""Podcast script generation controller."""

from __future__ import annotations

import os
import re

from gtts import gTTS

from .BaseController import BaseController
from ..Stores.LLM import GenAIProvider
from ..models.prompts import PodCastPromptEnum


class PodcastGenController(BaseController):
    def __init__(
        self,
        provider: GenAIProvider,
        lang: str = "en",
        topic: str = None,
        style: str = "Simple & Clear",
        duration: int = 3,
    ) -> None:
        super().__init__()

        self.lang = lang if lang != "auto" else "en"
        self.provider = provider
        self.topic = topic
        self.style = style
        self.duration = duration

        self.template = (
            PodCastPromptEnum.AR.value
            if self.lang.lower().strip() == "ar"
            else PodCastPromptEnum.EN.value
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
        _, intro_words, main_words, conclusion_words = self.calculate_words(words_per_minute)

        prompt = self.template.format(
            topic=self.topic,
            style=self.style,
            total_minutes=self.duration,
            intro_words=intro_words,
            main_words=main_words,
            conclusion_words=conclusion_words,
        )

        response = self.provider.generate_chunks(prompt, temperature=0.4)
        json_output = self.script_to_json(response) if response else {}
        return response, json_output

    # ── Parsing ───────────────────────────────────────────────────────────────

    def script_to_json(self, raw_script: str) -> dict:
        cleaned_script = re.sub(r"\*\*|\*", "", raw_script)
        sections: list[dict] = []
        current_section: dict | None = None

        for line in cleaned_script.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Section headers
            if line.startswith("[INTRO]") or line.startswith("[Q&A SESSION]") or line.startswith("[OUTRO]"):
                if current_section is not None:
                    sections.append(current_section)
                current_section = {"title": line.strip("[]"), "parts": []}
                continue

            if current_section is None:
                continue

            if line.startswith("[host]:"):
                current_section["parts"].append({
                    "type": "host",
                    "text": line.replace("[host]:", "").strip(),
                })
            elif line.startswith("[speaker_a]:"):
                current_section["parts"].append({
                    "type": "speaker_a",
                    "text": line.replace("[speaker_a]:", "").strip(),
                })

        # Append last open section
        if current_section is not None:
            sections.append(current_section)

        return {
            "topic": self.topic,
            "style": self.style,
            "duration": self.duration,
            "sections": sections,
        }

    # ── Audio ─────────────────────────────────────────────────────────────────

    def script_to_audio(self, script_text: str, lang: str = "en") -> str:
        """Convert podcast script text to an MP3 file and return the path."""
        os.makedirs(self.temp_dir, exist_ok=True)
        random_name = self.generate_random_string(8)
        filename = os.path.join(self.temp_dir, f"{random_name}_podcast.mp3")

        tts = gTTS(text=script_text, lang=lang, tld="com" if lang != "ar" else None)
        tts.save(filename)
        return filename