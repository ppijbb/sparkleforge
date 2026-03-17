"""Local subprocess sandbox backend (context_mode SandboxedExecutor)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.core.sandbox.backends.base import BaseSandboxBackend, ExecuteResponse

logger = logging.getLogger(__name__)


class LocalSandboxBackend(BaseSandboxBackend):
    """Run code in a local subprocess with minimal env (no network by default)."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._executor = None

    @property
    def id(self) -> str:
        return "local"

    def _get_executor(self):
        if self._executor is None:
            from src.core.context_mode.executor import SandboxedExecutor

            self._executor = SandboxedExecutor(project_root=str(self._project_root))
        return self._executor

    async def execute_code(self, code: str, language: str = "python") -> ExecuteResponse:
        """Run code via SandboxedExecutor in a thread (sync API)."""
        executor = self._get_executor()
        timeout_ms = 30_000
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: executor.execute(
                        language=language,
                        code=code,
                        timeout=timeout_ms,
                    ),
                ),
                timeout=timeout_ms / 1000 + 5,
            )
            out = (result.stdout or "") + ((result.stderr and ("\n" + result.stderr)) or "")
            return ExecuteResponse(
                output=out,
                exit_code=result.exit_code,
                truncated=result.timed_out,
            )
        except asyncio.TimeoutError:
            return ExecuteResponse(
                output="",
                exit_code=124,
                truncated=True,
                error="Execution timed out",
            )
        except Exception as e:
            logger.warning("Local sandbox execute_code failed: %s", e)
            return ExecuteResponse(
                output="",
                exit_code=1,
                truncated=False,
                error=str(e),
            )
