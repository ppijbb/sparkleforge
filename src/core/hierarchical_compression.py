"""Hierarchical compression module - thin wrapper over src.core.compression."""

from src.core.compression import (
    CompressionLevel,
    CompressionResult,
    CompressionValidator,
    CompressionHistory,
    HierarchicalCompressor,
    compress_data,
)

__all__ = [
    "CompressionLevel",
    "CompressionResult",
    "CompressionValidator",
    "CompressionHistory",
    "HierarchicalCompressor",
    "compress_data",
]
