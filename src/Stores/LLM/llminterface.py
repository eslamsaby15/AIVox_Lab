"""
Abstract interface that every LLM provider must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str) -> None:
        """Set the model used for text generation."""

    @abstractmethod
    def set_embedded_model(self, model_id: str, embedding_size: int) -> None:
        """Set the model used for text embedding."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        chat_history: Optional[List[dict]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """Generate a single-turn or multi-turn response."""

    @abstractmethod
    def generate_chunks(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """
        Generate a response for long-form / document tasks.
        Unlike ``generate_text``, this skips chat history and focuses
        on single large-prompt generation (e.g. scripts, summaries).
        """

    @abstractmethod
    def embed_text(
        self,
        text: str,
        document_type: Optional[str] = None,
    ) -> Optional[List[float]]:
        """Embed text and return a float vector."""

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str) -> dict:
        """Wrap a plain-text prompt in the provider's message format."""