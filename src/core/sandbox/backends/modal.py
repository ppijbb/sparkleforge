"""Optional Modal sandbox backend (langchain_modal).

Install: pip install langchain-modal.
Configure: MODAL_* env vars per langchain-modal docs.
"""

from __future__ import annotations

import logging

from src.core.sandbox.backends.base import BaseSandboxBackend, ExecuteResponse

logger = logging.getLogger(__name__)


class ModalSandboxBackend(BaseSandboxBackend):
    """Execute code in a Modal sandbox (remote serverless environment)."""

    def __init__(self) -> None:
        self._sandbox = None

    @property
    def id(self) -> str:
        return "modal"

    def _ensure_sandbox(self):
        if self._sandbox is not None:
            return
        try:
            from langchain_modal.sandbox import ModalSandbox
            self._sandbox = ModalSandbox()
        except ImportError as e:
            raise ImportError(
                "Modal sandbox requires langchain-modal: pip install langchain-modal"
            ) from e

    async def execute_code(self, code: str, language: str = "python") -> ExecuteResponse:
        """Run code in Modal sandbox."""
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
