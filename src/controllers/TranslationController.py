"""Translation controller — supports classical HuggingFace (EN→AR) and LLM (multi-language) modes."""

from __future__ import annotations

from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .BaseController import BaseController
from ..Stores.LLM.Providers.geminiProvider import GenAIProvider


class TranslationController(BaseController):
    def __init__(
        self,
        provider: Optional[GenAIProvider] = None,
        mode: str = "llm",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.provider = provider

        # Template supports any target language via {target_language}
        self.prompt_template = "\n\n".join([
            "You are a professional translator.",
            "## Rules:",
            "- Translate the text faithfully into {target_language}.",
            "- Do not change, add, or remove words.",
            "- Keep numbers, symbols, and punctuation unchanged.",
            "## Text:",
            "{text}",
            "## Translation:",
        ])

        self.prompt = PromptTemplate(
            input_variables=["target_language", "text"],
            template=self.prompt_template,
        )

    def classical_translator(self, text: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """Translate English → Arabic using HuggingFace pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            translator = hf_pipeline(
                "translation_en_to_ar",
                model=self.app_settings.EN_AR_MODEL,
                tokenizer=self.app_settings.EN_AR_MODEL,
                device=-1,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load translation model: {exc}") from exc

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_text(text)
        translations: list[str] = []

        for chunk in chunks:
            output = translator(chunk, max_length=512, do_sample=True, temperature=temperature)
            translations.append(output[0]["translation_text"])

        return "\n".join(f"- {s}" for s in translations)

    # Backward-compat alias
    def classical_Translator(self, text: str, **kwargs) -> str:  # noqa: N802
        return self.classical_translator(text, **kwargs)

    def llm_translation(
        self,
        text: str,
        target_language: str = "Arabic",
        max_length: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Translate text to any language using the configured LLM provider."""
        if not self.provider:
            raise RuntimeError("No provider configured for LLM Translation.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=max_length, chunk_overlap=50)
        chunks = splitter.split_text(text)
        if not chunks:
            raise ValueError("Input text is empty.")

        translations: list[str] = []
        for chunk in chunks:
            prompt = self.prompt_template.format(target_language=target_language, text=chunk)
            translated = self.provider.generate_text(
                prompt=prompt,
                max_output_tokens=max_length,
                temperature=temperature,
            )
            if translated:
                translations.append(translated.strip())

        return "\n".join(f"- {s}" for s in translations)

    # Backward-compat alias
    def LLM_Translation(self, text: str, **kwargs) -> str:  # noqa: N802
        return self.llm_translation(text, **kwargs)


# Backward-compat alias (old typo name)
TransaltionController = TranslationController
