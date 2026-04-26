"""Google Gemini LLM provider."""

from __future__ import annotations

from typing import List, Optional, Union

import google.generativeai as genai

from ..LLMEnums import GeminiEnums
from ..llminterface import LLMInterface
from ....helpers.logger import get_logger


class GenAIProvider(LLMInterface):
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
        genai.configure(api_key=self.api_key)

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
        return {"role": role, "parts": [self.process_text(prompt)]}

    # ── Generation ────────────────────────────────────────────────────────────

    def generate_text(
        self,
        prompt: str,
        chat_history: Optional[List[dict]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        if not self.gen_model_id:
            self.logger.error("Generation model for Gemini was not set.")
            return None

        max_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temp = temperature if temperature is not None else self.default_generation_temperature

        try:
            model = genai.GenerativeModel(self.gen_model_id)

            if chat_history:
                # Multi-turn: use Gemini chat session
                chat = model.start_chat(history=chat_history)
                response = chat.send_message(
                    self.process_text(prompt),
                    generation_config={
                        "temperature": temp,
                        "max_output_tokens": max_tokens,
                    },
                )
            else:
                response = model.generate_content(
                    self.process_text(prompt),
                    generation_config={
                        "temperature": temp,
                        "max_output_tokens": max_tokens,
                    },
                )
        except Exception as exc:
            self.logger.error("Gemini API error: %s", exc)
            return None

        if not response or not response.text:
            self.logger.error("Empty response from Gemini.")
            return None

        return response.text

    def generate_chunks(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """Standalone large-prompt generation without chat history."""
        if not self.gen_model_id:
            self.logger.error("Generation model for Gemini was not set.")
            return None

        temp = temperature or self.default_generation_temperature

        try:
            model = genai.GenerativeModel(self.gen_model_id)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": temp},
            )
        except Exception as exc:
            self.logger.error("Gemini API error: %s", exc)
            return None

        if not response or not response.text:
            self.logger.error("Empty response from Gemini.")
            return None

        return response.text

    # Backward-compat alias
    def generate_Chunks(self, prompt: str, temperature: float = 0.3) -> Optional[str]:  # noqa: N802
        return self.generate_chunks(prompt, temperature)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_text(
        self,
        text: Union[str, List[str]],
        document_type: Optional[str] = None,
    ) -> Optional[List[float]]:
        if not self.embedd_model_id:
            self.logger.error("Embedding model for Gemini was not set.")
            return None

        if isinstance(text, str):
            text = [text]

        task_type = GeminiEnums.DOCUMENT.value
        if document_type == GeminiEnums.QUERY.value:
            task_type = GeminiEnums.QUERY.value

        try:
            response = genai.embed_content(
                model=self.embedd_model_id,
                content=text,
                task_type=task_type,
            )
        except Exception as exc:
            self.logger.error("Gemini embedding error: %s", exc)
            return None

        if not response or "embedding" not in response:
            self.logger.error("Empty embedding response from Gemini.")
            return None

        return response["embedding"]

    # Legacy alias
    def embedd_text(self, text, document_type=None):
        return self.embed_text(text, document_type)
