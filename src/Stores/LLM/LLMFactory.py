"""
LLM Provider Factory.

Usage::

    factory = LLMProviderFactory(get_settings())
    provider = factory.create("Gemini")
    provider.set_generation_model(settings.GENERATION_MODEL_ID_GEMINI)
"""

from __future__ import annotations

from typing import Optional

from .Providers import CohereProvider, GenAIProvider, OpenAiProvider
from .LLMEnums import LLMEnums
from .llminterface import LLMInterface
from ...helpers.config import Settings
from ...helpers.logger import get_logger

logger = get_logger(__name__)


class LLMProviderFactory:
    def __init__(self, config: Settings) -> None:
        self.config = config

    def create(self, provider: str) -> Optional[LLMInterface]:
        """Instantiate and return the provider for the given backend name."""

        if provider == LLMEnums.OPENAI.value:
            return OpenAiProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
            )

        if provider == LLMEnums.COHERE.value:
            return CohereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
            )

        if provider == LLMEnums.GEMINI.value:
            return GenAIProvider(
                api_key=self.config.GEMINI_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE,
            )

        logger.warning("Unknown LLM provider requested: '%s'", provider)
        return None