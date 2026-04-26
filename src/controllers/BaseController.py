"""
Base controller — shared utilities for all feature controllers.
"""

from __future__ import annotations

import os
import random
import string

from ..helpers.config import get_settings
from ..helpers.logger import get_logger


class BaseController:
    def __init__(self) -> None:
        self.app_settings = get_settings()
        # Resolve paths relative to src/
        self.base_dir = os.path.dirname(os.path.dirname(__file__))  # src/
        self.files_dir = os.path.join(self.base_dir, "assets", "Data")
        self.temp_dir = os.path.join(self.base_dir, "assets", "temp")

        # Ensure storage directories exist on first use
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.logger = get_logger(self.__class__.__name__)

        # Legacy alias kept for backward-compatibility with existing code
        self.app_setting = self.app_settings

    def generate_random_string(self, length: int = 6) -> str:
        """Return a random lowercase alphanumeric string of *length* chars."""
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
