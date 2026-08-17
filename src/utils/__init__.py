"""Utilities Package

This package contains utility modules for configuration management,
logging, and other common functionality.
"""

import random
import time

from .logger import get_logger, quick_logger, setup_logger


def JitteredBackoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Return a jittered exponential backoff delay in seconds.

    This standardises the rate-limiting recovery backoff calculation across
    all model request clients (see issue #660). ``attempt`` is the zero-based
    retry index. The delay grows exponentially from ``base_delay`` up to
    ``max_delay`` with full jitter to avoid thundering-herd retries.
    """
    if attempt < 0:
        attempt = 0
    if base_delay <= 0:
        base_delay = 1.0
    if max_delay <= 0:
        max_delay = 60.0
    exponential = min(max_delay, base_delay * (2 ** attempt))
    return random.uniform(0, exponential)


__all__ = ["setup_logger", "get_logger", "quick_logger", "JitteredBackoff"]
