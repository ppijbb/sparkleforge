import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    RETRYABLE = "retryable"  # e.g. Rate limit, Timeouts
    MODEL_FAILURE = "model_failure"  # e.g. Content filter, invalid params
    TOOL_FAILURE = "tool_failure"  # e.g. Tool crashed or returned error
    CONTEXT_LIMIT = "context_limit"  # Token limit exceeded
    FATAL = "fatal"  # Authentication, Permissions, etc.


class ErrorClassifier:
    """Intelligent Error Classifier for SparkleForge (Phase 4)."""

    @staticmethod
    def classify(exception: Exception) -> ErrorCategory:
        error_msg = str(exception).lower()

        # Rate limits (모든 provider가 일시 소진된 경우 포함 — 대기 후 회복 가능)
        if any(
            kw in error_msg
            for kw in [
                "rate limit",
                "429",
                "too many requests",
                "all fallback models failed",
                "no available models",
            ]
        ):
            return ErrorCategory.RETRYABLE

        # Timeouts
        if any(kw in error_msg for kw in ["timeout", "timed out", "deadline exceeded"]):
            return ErrorCategory.RETRYABLE

        # Context limits
        if any(kw in error_msg for kw in ["context length", "token limit", "maximum context"]):
            return ErrorCategory.CONTEXT_LIMIT

        # Authentication (Fatal)
        if any(kw in error_msg for kw in ["auth", "api key", "unauthorized", "401", "403"]):
            return ErrorCategory.FATAL

        # Default to fatal or treat as general model failure
        return ErrorCategory.MODEL_FAILURE
