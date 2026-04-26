"""Topic tagging controller."""

from __future__ import annotations

import re
from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .BaseController import BaseController
from ..Stores.LLM.Providers.geminiProvider import GenAIProvider


class TopicTaggingController(BaseController):
    def __init__(self, provider: Optional[GenAIProvider] = None) -> None:
        super().__init__()
        self.provider = provider

        self.prompt_template = "\n".join([
            "You are a helpful assistant that extracts topic tags from text.",
            "Read the following text and generate a list of relevant topic tags or keywords.",
            "- Focus on main subjects, concepts, named entities, or recurring themes.",
            "- Return tags as a comma-separated list only (no bullet points, no numbering).",
            "",
            "Text:",
            "{text}",
            "",
            "Tags:",
        ])

        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.prompt_template,
        )

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    def extract_tags(self, text: str, max_length: int = 300) -> list[str]:
        """Analyze text in chunks and return a deduplicated list of topic tags."""
        if not self.provider:
            raise RuntimeError("No provider configured for topic tagging.")

        chunks = self.splitter.split_text(text)
        if not chunks:
            raise ValueError("Input text is empty.")

        all_tags: set[str] = set()
        for chunk in chunks:
            prompt = self.prompt_template.format(text=chunk)
            try:
                output = self.provider.generate_text(
                    prompt=prompt,
                    max_output_tokens=max_length,
                    temperature=0.3,
                )
            except Exception as exc:
                self.logger.error("Topic tagging error: %s", exc)
                continue

            if not output:
                continue

            tags = [
                # Strip stray markdown or numbering
                re.sub(r"^[\s\-\*\d\.]+", "", tag).strip()
                for tag in output.split(",")
                if tag.strip()
            ]
            all_tags.update(t for t in tags if t)

        return sorted(all_tags)
