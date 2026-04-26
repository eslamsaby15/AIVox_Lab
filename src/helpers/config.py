"""
Centralised application settings.

Reads from .env (or environment variables).
Call ``get_settings()`` everywhere — it returns a cached singleton so
the .env file is only parsed once per process.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings

# Resolve .env path relative to this file (src/helpers/ → src/ → project root)
_SRC_DIR = os.path.dirname(os.path.dirname(__file__))  # src/
_ROOT_DIR = os.path.dirname(_SRC_DIR)                   # project root
_ENV_PATH = os.path.join(_ROOT_DIR, ".env")


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "AIVox Lab"
    APP_VERSION: str = "1.0.0"
    APP_DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── HuggingFace / Classical models ────────────────────────────────────────
    EN_MODEL: str = "facebook/bart-large-cnn"
    EN_AR_MODEL: str = "Helsinki-NLP/opus-mt-en-ar"

    # ── LLM backend identifiers ───────────────────────────────────────────────
    GENERATION_BACKEND_OPENAI: str = "OpenAI"
    GENERATION_BACKEND_GEMINI: str = "Gemini"
    GENERATION_BACKEND_COHERE: str = "Cohere"

    EMBEDDING_BACKEND_GEMINI: str = "Gemini"
    EMBEDDING_BACKEND: str = "Cohere"

    # ── API keys (optional — will be None if not set) ─────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # ── Generation model IDs ──────────────────────────────────────────────────
    GENERATION_MODEL_ID_OPENAI: Optional[str] = "gpt-4o-mini"
    GENERATION_MODEL_ID_GEMINI: Optional[str] = "gemini-2.0-flash"
    GENERATION_MODEL_ID_COHERE_LIGHT: Optional[str] = "command-r-08-2024"

    # ── Embedding model IDs ───────────────────────────────────────────────────
    EMBEDDING_MODEL_ID: Optional[str] = "text-embedding-3-small"
    EMBEDDING_MODEL_ID_GEMINI: Optional[str] = "models/embedding-001"
    EMBEDDING_MODEL_ID_COHERE_MULTILINGUAL: Optional[str] = "embed-multilingual-v3.0"

    # ── API URLs ──────────────────────────────────────────────────────────────
    OPENAI_API_URL: str = "https://api.openai.com/v1"

    # ── Generation defaults ───────────────────────────────────────────────────
    INPUT_DEFAULT_MAX_CHARACTERS: int = 4096
    GENERATION_DEFAULT_MAX_TOKENS: int = 1024
    GENERATION_DEFAULT_TEMPERATURE: float = 0.3

    # ── Storage paths ─────────────────────────────────────────────────────────
    TEMP_DIR: str = "src/assets/temp"
    DATA_DIR: str = "src/assets/Data"

    # ── Backward-compat aliases (old typo names) ──────────────────────────────
    @property
    def INPUT_DAFAULT_MAX_CHARACTERS(self) -> int:  # noqa: N802
        return self.INPUT_DEFAULT_MAX_CHARACTERS

    @property
    def GENERATION_DAFAULT_MAX_TOKENS(self) -> int:  # noqa: N802
        return self.GENERATION_DEFAULT_MAX_TOKENS

    @property
    def GENERATION_DAFAULT_TEMPERATURE(self) -> float:  # noqa: N802
        return self.GENERATION_DEFAULT_TEMPERATURE

    class Config:
        env_file = _ENV_PATH
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()


# Legacy alias — older code calls APP_Setting()
def APP_Setting() -> Settings:  # noqa: N802
    return get_settings()


def validate_provider_keys(provider: str) -> tuple[bool, str]:
    """
    Check that the API key for the given provider is configured.

    Returns (ok: bool, message: str).
    """
    s = get_settings()
    checks = {
        "OpenAI": (s.OPENAI_API_KEY, "OPENAI_API_KEY"),
        "Gemini": (s.GEMINI_API_KEY, "GEMINI_API_KEY"),
        "Cohere": (s.COHERE_API_KEY, "COHERE_API_KEY"),
    }
    key_value, env_name = checks.get(provider, (None, provider))
    if not key_value or key_value.startswith("YOUR_"):
        return False, f"⚠️ {env_name} is not set in your .env file."
    return True, "OK"
