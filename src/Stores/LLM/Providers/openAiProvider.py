"""OpenAI LLM provider."""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from ..LLMEnums import LLMEnums, OpenEnums
from ..llminterface import LLMInterface
from ....helpers.logger import get_logger


class OpenAiProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        api_url: str,
        default_input_max_characters: int = 4096,
        default_generation_max_output_tokens: int = 1024,
        default_generation_temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.gen_model_id: Optional[str] = None
        self.embedd_model_id: Optional[str] = None
        self.embedding_size: Optional[int] = None
        self.OpenEnums = OpenEnums

        self.logger = get_logger(__name__)
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)

    # ── Model configuration ──────────────────────────────────────────────────

    def set_generation_model(self, model_id: str) -> None:
        self.gen_model_id = model_id

    def set_embedded_model(self, model_id: str, embedding_size: int) -> None:
        self.embedd_model_id = model_id
        self.embedding_size = embedding_size

    # ── Text helpers ─────────────────────────────────────────────────────────

    def process_text(self, text: str) -> str:
        return text[: self.default_input_max_characters].strip()

    def construct_prompt(self, prompt: str, role: str) -> dict:
        return {"role": role, "content": self.process_text(prompt)}

    # ── Generation ───────────────────────────────────────────────────────────

    def generate_text(
        self,
        prompt: str,
        chat_history: Optional[List[dict]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.3,
    ) -> Optional[str]:
        if not self.client:
            self.logger.error("OpenAI client was not initialised.")
            return None
        if not self.gen_model_id:
            self.logger.error("Generation model for OpenAI was not set.")
            return None

        history = list(chat_history) if chat_history else []
        history.append(self.construct_prompt(prompt, role=self.OpenEnums.USER.value))

        max_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temp = temperature or self.default_generation_temperature

        try:
            response = self.client.chat.completions.create(
                model=self.gen_model_id,
                messages=history,
                max_tokens=max_tokens,
                temperature=temp,
            )
        except Exception as exc:
            self.logger.error("OpenAI API error: %s", exc)
            return None

        if (
            not response
            or not response.choices
            or not response.choices[0].message
        ):
            self.logger.error("Empty response from OpenAI.")
            return None

        # SDK v1 returns an object, not a dict
        return response.choices[0].message.content

    def generate_chunks(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """Generate a standalone response (no chat history)."""
        if not self.client:
            self.logger.error("OpenAI client was not initialised.")
            return None
        if not self.gen_model_id:
            self.logger.error("Generation model for OpenAI was not set.")
            return None

        messages = [
            self.construct_prompt(prompt, role=self.OpenEnums.SYSTEM.value)
        ]
        temp = temperature or self.default_generation_temperature

        try:
            response = self.client.chat.completions.create(
                model=self.gen_model_id,
                messages=messages,
                temperature=temp,
            )
        except Exception as exc:
            self.logger.error("OpenAI API error: %s", exc)
            return None

        if (
            not response
            or not response.choices
            or not response.choices[0].message
        ):
            self.logger.error("Empty response from OpenAI.")
            return None

        return response.choices[0].message.content

    # ── Backward compat alias ─────────────────────────────────────────────────
    def generate_Chunks(self, prompt: str, temperature: float = 0.3) -> Optional[str]:  # noqa: N802
        return self.generate_chunks(prompt, temperature)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_text(
        self, text: str, document_type: Optional[str] = None
    ) -> Optional[List[float]]:
        if not self.client:
            self.logger.error("OpenAI client was not initialised.")
            return None
        if not self.embedd_model_id:
            self.logger.error("Embedding model for OpenAI was not set.")
            return None

        try:
            response = self.client.embeddings.create(
                model=self.embedd_model_id, input=text
            )
        except Exception as exc:
            self.logger.error("OpenAI embedding error: %s", exc)
            return None

        if not response or not response.data or not response.data[0].embedding:
            self.logger.error("Empty embedding response from OpenAI.")
            return None

        return response.data[0].embedding

    # Legacy alias
    def embedd_text(self, text: str, document_type: Optional[str] = None):
        return self.embed_text(text, document_type)