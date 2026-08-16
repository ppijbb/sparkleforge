import os
from typing import List, Optional


class ConfigurationError(Exception):
    """Raised when the environment configuration is invalid or missing."""
    pass


class EnvConfigurator:
    def __init__(self, environ: Optional[dict] = None):
        self.environ = environ if environ is not None else os.environ

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.environ.get(key, default)

    def verify(self, required_vars: Optional[List[str]] = None) -> bool:
        if required_vars is None:
            self.environ.setdefault("SPARKLEFORGE_ENV", "development")
            required_vars = ["SPARKLEFORGE_ENV"]

        missing = [var for var in required_vars if var not in self.environ]
        if missing:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        return True


def verify_environment(required_vars: List[str] = None) -> bool:
    """Legacy wrapper for EnvConfigurator.verify."""
    configurator = EnvConfigurator()
    return configurator.verify(required_vars)
