import os
from typing import List, Optional


class ConfigurationError(Exception):
    """Raised when the environment configuration is invalid or missing."""
    pass


class EnvConfigurator:
    def __init__(self, environ: Optional[dict] = None):
        self.environ = environ if environ is not None else os.environ

    def verify_environment(self, required_vars: List[str] = None) -> bool:
        """
        Validates that the environment is correctly configured.

        Checks for the presence of required environment variables.
        A variable is considered present when it exists in the environment,
        even if its value is an empty string.
        """
        if required_vars is None:
            required_vars = ["SPARKLEFORGE_ENV"]

        missing = [var for var in required_vars if var not in self.environ]

        if missing:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return True


def verify_environment(required_vars: List[str] = None) -> bool:
    """
    Legacy wrapper for EnvConfigurator.verify_environment.
    """
    return EnvConfigurator().verify_environment(required_vars)
