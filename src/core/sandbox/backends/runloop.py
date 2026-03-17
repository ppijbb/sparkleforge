"""Optional Runloop sandbox backend (langchain_runloop / runloop-api-client).

Install: pip install langchain-runloop (or runloop-api-client).
Configure: RUNLOOP_API_KEY, RUNLOOP_DEVELOPMENT_BOX_ID or create devbox at runtime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.sandbox.backends.base import BaseSandboxBackend, ExecuteResponse

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RunloopSandboxBackend(BaseSandboxBackend):
    """Execute code on a Runloop devbox (remote isolated environment)."""

    def __init__(self) -> None:
        self._devbox = None
        self._client = None

    @property
    def id(self) -> str:
        return "runloop"

    def _ensure_devbox(self):
        if self._devbox is not None:
            return
        try:
            from runloop_api_client.sdk import Runloop
            import os
            api_key = os.environ.get("RUNLOOP_API_KEY")
            devbox_id = os.environ.get("RUNLOOP_DEVELOPMENT_BOX_ID")
            if not api_key:
                raise RuntimeError("RUNLOOP_API_KEY not set")
            client = Runloop(api_key=api_key)
            if devbox_id:
                self._devbox = client.development_boxes.get(devbox_id)
            else:
                devboxes = list(client.development_boxes.list())
                if not devboxes:
                    raise RuntimeError("No Runloop devbox found; create one or set RUNLOOP_DEVELOPMENT_BOX_ID")
                self._devbox = devboxes[0]
            self._client = client
        except ImportError as e:
            raise ImportError("Runloop sandbox requires runloop-api-client: pip install runloop-api-client") from e

    async def execute_code(self, code: str, language: str = "python") -> ExecuteResponse:
        """Run code on the Runloop devbox via shell command."""
        self._ensure_devbox()
        if language.lower() in ("python", "py"):
            command = f"python3 -c {repr(code)}"
        else:
            command = code
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._devbox.cmd.exec(command, timeout=60 * 30),
        )
        out = (result.stdout() or "") + ((result.stderr() and ("\n" + result.stderr())) or "")
        return ExecuteResponse(
            output=out,
            exit_code=result.exit_code,
            truncated=False,
        )
