"""Utilities Package

This package contains utility modules for configuration management,
logging, and other common functionality.
"""

from .logger import get_logger, quick_logger, setup_logger

__all__ = ["setup_logger", "get_logger", "quick_logger", "jittered_backoff"]

import random

def jittered_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff with full jitter.
    delay = min(max_delay, base_delay * 2^attempt)
    jittered_delay = random.uniform(0, delay)
    """
    delay = min(max_delay, base_delay * (2**attempt))
    return random.uniform(0, delay)

__all__ = ["setup_logger", "get_logger", "quick_logger"]
