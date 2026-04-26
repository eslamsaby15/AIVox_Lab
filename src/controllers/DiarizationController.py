"""Speaker diarization controller."""

from __future__ import annotations

import json
import time

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..models import DiarizationEnum, DiarizationResult, Segment
from .BaseController import BaseController
from ..Stores.LLM import GenAIProvider


class Diarization(BaseController):
    def __init__(
        self,
        lang: str,
        provider: GenAIProvider,
        chunk_size: int = 6000,
    ) -> None:
        super().__init__()
        self.lang = lang
        self.provider = provider
        self.chunk_size = chunk_size

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=100,
            length_function=len,
        )

        system_text = (
            DiarizationEnum.AR.value
            if self.lang.lower().strip() == "ar"
            else DiarizationEnum.EN.value
        )

        # Build the prompt template once
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_text),
            ("user", "{text}"),
        ])

    def run_diarization(self, transcript: str) -> DiarizationResult:
        chunks = self.splitter.split_text(transcript)
        all_segments: list[Segment] = []

        for i, chunk in enumerate(chunks):
            prompt = self.prompt_template.format(text=chunk)
            output = self.provider.generate_chunks(prompt)

            if not output:
                self.logger.warning("Empty output for chunk %d — skipping.", i)
                continue

            cleaned_text = self._clean_output(output)

            try:
                data = json.loads(cleaned_text)
                segments = [
                    Segment(speaker=turn["speaker"], text=turn["text"])
                    for turn in data.get("conversation", [])
                ]
                all_segments.extend(segments)
            except json.JSONDecodeError as exc:
                self.logger.warning("JSON parse error on chunk %d: %s", i, exc)
                continue

            # Respect rate limits between chunks
            if i < len(chunks) - 1:
                time.sleep(2)

        return DiarizationResult(segments=all_segments)

    def _clean_output(self, text: str) -> str:
        """Strip markdown code-fence wrappers from LLM JSON output."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text.strip()
