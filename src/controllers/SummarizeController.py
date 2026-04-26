"""Summarization controller — supports classical HuggingFace and LLM modes."""

from __future__ import annotations

from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .BaseController import BaseController
from ..Stores.LLM.Providers.geminiProvider import GenAIProvider


class SummarizeController(BaseController):
    def __init__(
        self,
        lang: str = "en",
        provider: Optional[GenAIProvider] = None,
        mode: str = "llm",
    ) -> None:
        super().__init__()
        self.lang = lang
        self.mode = mode
        self.provider = provider

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=50,
        )

        self.prompt_template = "\n\n".join([
            "You are a helpful assistant that summarizes text in {language}.",
            "Given the following text, generate:",
            "1. A concise summary.",
            "2. Important keywords.",
            "",
            "Text:",
            "{text}",
            "",
            "Output format:",
            "Summary:\n- <summary_here>",
            "",
            "Keywords:",
            "- <keyword1>, <keyword2>, ...",
        ])

        self.prompt = PromptTemplate(
            input_variables=["language", "text"],
            template=self.prompt_template,
        )

    def classical_summarizer(self, text: str) -> str:
        """Summarize using a classical HuggingFace pipeline (bart-large-cnn)."""
        try:
            from transformers import pipeline as hf_pipeline
            summarizer = hf_pipeline(
                "summarization",
                model=self.app_settings.EN_MODEL,
                tokenizer=self.app_settings.EN_MODEL,
                device=-1,  # CPU; use 0 for GPU if available
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load summarisation model: {exc}") from exc

        chunks = self.splitter.split_text(text)
        summaries: list[str] = []

        for chunk in chunks:
            output = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(output[0]["summary_text"])

        return "\n".join(f"- {s}" for s in summaries)

    # Backward-compat alias
    def classical_Summarizer(self, text: str) -> str:  # noqa: N802
        return self.classical_summarizer(text)

    def llm_summarizer(self, text: str, max_length: int = 500) -> str:
        """Summarise using the configured LLM provider."""
        if not self.provider:
            raise RuntimeError("No provider configured for LLM summarization.")

        chunks = self.splitter.split_text(text)
        if not chunks:
            raise ValueError("Input text is empty.")

        summaries: list[str] = []
        for chunk in chunks:
            prompt = self.prompt_template.format(language=self.lang, text=chunk)
            summary = self.provider.generate_chunks(prompt=prompt, temperature=0.3)
            if summary:
                summaries.append(summary.strip())

        return "\n".join(f"- {s}" for s in summaries)

    # Backward-compat alias
    def LLM_Summarizer(self, text: str, max_length: int = 500) -> str:  # noqa: N802
        return self.llm_summarizer(text, max_length)


# Backward-compat alias (old class name)
Summarizer = SummarizeController
