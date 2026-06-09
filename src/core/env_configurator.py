import os
from typing import List

class ConfigurationError(Exception):
    """Raised when the environment configuration is invalid or missing."""
    pass

def verify_environment(required_vars: List[str] = None) -> bool:
    """
    Validates that the environment is correctly configured.

    Checks for the presence of required environment variables. If any are missing,
    raises a ConfigurationError.

    Args:
        required_vars: A list of environment variable names that must be set.

    Returns:
        bool: True if validation passes.

    Raises:
        ConfigurationError: If any required environment variable is missing.
    """
    if required_vars is None:
        required_vars = ["SPARKLEFORGE_ENV"]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return True
