"""Base types for sandbox backends (Runloop, Daytona, Modal, local)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecuteResponse:
    """Result of sandboxed code/command execution."""

    output: str
    exit_code: int
    truncated: bool = False
    error: str | None = None


class BaseSandboxBackend(ABC):
    """Abstract base for isolated code execution backends."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Backend identifier (e.g. 'local', 'runloop', 'daytona')."""
        ...

    @abstractmethod
    async def execute_code(self, code: str, language: str = "python") -> ExecuteResponse:
        """Execute code in the sandbox. Returns combined stdout/stderr and exit code."""
        ...
