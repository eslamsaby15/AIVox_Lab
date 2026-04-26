"""Sentiment analysis controller."""

from __future__ import annotations

import re
from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .BaseController import BaseController
from ..Stores.LLM.Providers.geminiProvider import GenAIProvider


class SentimentAnalysisController(BaseController):
    def __init__(self, provider: Optional[GenAIProvider] = None) -> None:
        super().__init__()
        self.provider = provider

        self.prompt_template = "\n".join([
            "You are a helpful assistant that analyzes text sentiment.",
            "Read the following text and provide:",
            "- The main sentiment (Positive, Negative, or Neutral).",
            "- Key emotional points or themes found in the text.",
            "",
            "Text:",
            "{text}",
            "",
            "Output format (follow exactly):",
            "Sentiment: <Positive|Negative|Neutral>",
            "Key points: <point1>, <point2>, ...",
        ])

        self.prompt = PromptTemplate(
            input_variables=["text"],
            template=self.prompt_template,
        )

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    def analysis(self, text: str, max_length: int = 500) -> list[dict]:
        """Analyze text in chunks and return structured sentiment results."""
        if not self.provider:
            raise RuntimeError("No provider configured for sentiment analysis.")

        chunks = self.splitter.split_text(text)
        if not chunks:
            raise ValueError("Input text is empty.")

        results: list[dict] = []
        for chunk in chunks:
            prompt = self.prompt_template.format(text=chunk)

            try:
                output = self.provider.generate_text(
                    prompt=prompt,
                    max_output_tokens=max_length,
                    temperature=0.3,
                )
            except Exception as exc:
                self.logger.error("Sentiment analysis error: %s", exc)
                continue

            if not output:
                continue

            output = output.strip()
            sentiment_match = re.search(r"Sentiment:\s*(\w+)", output, re.IGNORECASE)
            keypoints_match = re.search(r"Key points:\s*(.*)", output, re.IGNORECASE | re.DOTALL)

            sentiment = sentiment_match.group(1).capitalize() if sentiment_match else "N/A"
            key_points = keypoints_match.group(1).strip() if keypoints_match else ""

            results.append({
                "chunk": chunk,
                "sentiment": sentiment,
                "key_points": key_points,
            })

        return results
