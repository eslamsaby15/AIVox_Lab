"""Cohere LLM provider (Cohere SDK v5 / ClientV2)."""

from __future__ import annotations

from typing import List, Optional, Union

import cohere

from ..LLMEnums import CoHereEnums, DocumentTypeEnum, LLMEnums
from ..llminterface import LLMInterface
from ....helpers.logger import get_logger


class CohereProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 4096,
        default_generation_max_output_tokens: int = 1024,
        default_generation_temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.gen_model_id: Optional[str] = None
        self.embedd_model_id: Optional[str] = None
        self.embedding_size: Optional[int] = None

        self.logger = get_logger(__name__)

        # Cohere SDK v5 uses ClientV2
        self.client = cohere.ClientV2(api_key=self.api_key)

    # ── Model configuration ───────────────────────────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        self.gen_model_id = model_id

    def set_embedded_model(self, model_id: str, embedding_size: int) -> None:
        self.embedd_model_id = model_id
        self.embedding_size = embedding_size

    # ── Text helpers ──────────────────────────────────────────────────────────

    def process_text(self, text: str) -> str:
        return text[: self.default_input_max_characters].strip()

    def construct_prompt(self, prompt: str, role: str) -> dict:
        return {"role": role, "content": self.process_text(prompt)}

    # ── Generation ────────────────────────────────────────────────────────────

    def generate_text(
        self,
        prompt: str,
        chat_history: Optional[List[dict]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.3,
    ) -> Optional[str]:
        if not self.client:
            self.logger.error("Cohere client was not initialised.")
            return None
        if not self.gen_model_id:
            self.logger.error("Generation model for Cohere was not set.")
            return None

        # Build messages list (ClientV2 style)
        messages: List[dict] = list(chat_history) if chat_history else []
        messages.append(self.construct_prompt(prompt, role=CoHereEnums.USER.value))

        max_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temp = temperature or self.default_generation_temperature

        try:
            response = self.client.chat(
                model=self.gen_model_id,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self.logger.error("Cohere API error: %s", exc)
            return None

        if not response or not response.message or not response.message.content:
            self.logger.error("Empty response from Cohere.")
            return None

        return response.message.content[0].text

    def generate_chunks(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """Standalone generation without chat history."""
        if not self.client:
            self.logger.error("Cohere client was not initialised.")
            return None
        if not self.gen_model_id:
            self.logger.error("Generation model for Cohere was not set.")
            return None

        temp = temperature or self.default_generation_temperature

        try:
            response = self.client.chat(
                model=self.gen_model_id,
                messages=[{"role": "user", "content": self.process_text(prompt)}],
                temperature=temp,
            )
        except Exception as exc:
            self.logger.error("Cohere API error: %s", exc)
            return None

        if not response or not response.message or not response.message.content:
            self.logger.error("Empty response from Cohere.")
            return None

        return response.message.content[0].text

    # Backward-compat alias
    def generate_Chunks(self, prompt: str, temperature: float = 0.3) -> Optional[str]:  # noqa: N802
        return self.generate_chunks(prompt, temperature)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_text(
        self,
        text: str,
        document_type: Optional[str] = None,
    ) -> Optional[List[float]]:
        if not self.embedd_model_id:
            self.logger.error("Embedding model for Cohere was not set.")
            return None

        input_type = CoHereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = CoHereEnums.QUERY.value

        try:
            response = self.client.embed(
                model=self.embedd_model_id,
                texts=[self.process_text(text)],
                input_type=input_type,
                embedding_types=["float"],
            )
        except Exception as exc:
            self.logger.error("Cohere embedding error: %s", exc)
            return None

        if (
            not response
            or not response.embeddings
            or not response.embeddings.float_
        ):
            self.logger.error("Empty embedding response from Cohere.")
            return None

        return response.embeddings.float_[0]

    # Legacy alias
    def embedd_text(self, text: str, document_type: Optional[str] = None):
        return self.embed_text(text, document_type)