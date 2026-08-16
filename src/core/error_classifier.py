import logging
from src.core.exceptions import ErrorCategory as BaseErrorCategory

logger = logging.getLogger(__name__)


class ErrorClassifier:
    """Intelligent Error Classifier for SparkleForge (Phase 4)."""

    @staticmethod
    def classify(exception: Exception) -> BaseErrorCategory:
        error_msg = str(exception).lower()

        if any(kw in error_msg for kw in ["rate limit", "429", "too many requests"]):
            return BaseErrorCategory.NETWORK
        if any(kw in error_msg for kw in ["timeout", "timed out", "deadline exceeded"]):
            return BaseErrorCategory.NETWORK
        if any(kw in error_msg for kw in ["auth", "api key", "unauthorized", "401", "403"]):
            return BaseErrorCategory.AUTHENTICATION
        if any(kw in error_msg for kw in ["context length", "token limit"]):
            return BaseErrorCategory.RESOURCE

        return BaseErrorCategory.SYSTEM
