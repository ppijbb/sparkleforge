import os
from typing import List


class ConfigurationError(Exception):
    """Raised when the environment configuration is invalid or missing."""

    pass


def verify_environment(required_vars: List[str] = None) -> bool:
    """
    Validates that the environment is correctly configured.

    Checks for the presence of required environment variables after any
    environment population logic has run. A variable is considered present when
    it exists in os.environ, even if its value is an empty string.

    Args:
        required_vars: A list of environment variable names that must be set.

    Returns:
        bool: True if validation passes.

    Raises:
        ConfigurationError: If any required environment variable is missing.
    """
    if required_vars is None:
        required_vars = ["SPARKLEFORGE_ENV"]

    missing = [var for var in required_vars if var not in os.environ]

    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return True
