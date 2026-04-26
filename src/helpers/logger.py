"""
Centralised logging configuration for AIVox Lab.

Usage::

    from src.helpers.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello!")
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with a consistent format.

    The first call configures the root handler; subsequent calls just
    return the named child logger (logging is hierarchical).
    """
    root = logging.getLogger()

    if not root.handlers:
        # Configure once
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)

    # Honour LOG_LEVEL from env without importing full Settings (avoids circular)
    import os
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    return logging.getLogger(name)
