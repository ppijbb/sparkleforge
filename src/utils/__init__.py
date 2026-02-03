"""
Utilities Package

This package contains utility modules for configuration management,
logging, and other common functionality.
"""

from .logger import setup_logger, get_logger, quick_logger

__all__ = [
    'setup_logger',
    'get_logger',
    'quick_logger'
]
