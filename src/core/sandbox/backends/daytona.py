"""Optional Daytona sandbox backend (langchain_daytona).

Install: pip install langchain-daytona.
Configure: DAYTONA_* env vars per langchain-daytona docs.
"""

from __future__ import annotations

import logging

from src.core.sandbox.backends.base import BaseSandboxBackend, ExecuteResponse

logger = logging.getLogger(__name__)


class DaytonaSandboxBackend(BaseSandboxBackend):
    """Execute code in a Daytona sandbox (remote isolated environment)."""

    def __init__(self) -> None:
        self._sandbox = None

    @property
    def id(self) -> str:
        return "daytona"

    def _ensure_sandbox(self):
        if self._sandbox is not None:
            return
        try:
            from langchain_daytona.sandbox import DaytonaSandbox
            self._sandbox = DaytonaSandbox()
        except ImportError as e:
            raise ImportError(
                "Daytona sandbox requires langchain-daytona: pip install langchain-daytona"
            ) from e

    async def execute_code(self, code: str, language: str = "python") -> ExecuteResponse:
        """Run code in Daytona sandbox (shell command)."""
        self._ensure_sandbox()
        if language.lower() in ("python", "py"):
            command = f"python3 -c {repr(code)}"
        else:
            command = code
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._sandbox.execute(command, timeout=30 * 60),
        )
        return ExecuteResponse(
            output=result.output,
            exit_code=result.exit_code,
            truncated=getattr(result, "truncated", False),
        )
