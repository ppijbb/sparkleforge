"""Sandbox backend factory: select local, docker, era, runloop, daytona, or modal from env."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from src.core.sandbox.backends.base import BaseSandboxBackend
from src.core.sandbox.backends.local import LocalSandboxBackend

logger = logging.getLogger(__name__)

_SANDBOX_BACKEND: Optional[BaseSandboxBackend] = None


def get_sandbox_backend(project_root: Optional[Path] = None) -> Optional[BaseSandboxBackend]:
    """Return the configured sandbox backend for code execution.

    Env SANDBOX_BACKEND: local | docker | era | runloop | daytona | modal.
    - local: subprocess via context_mode SandboxedExecutor (default).
    - docker: existing DockerSandbox (handled in _execute_code_tool).
    - era: existing ERA client (handled in _execute_code_tool).
    - runloop: optional langchain_runloop devbox (if installed and RUNLOOP_* set).
    - daytona: optional langchain_daytona (if installed and DAYTONA_* set).
    - modal: optional langchain_modal (if installed and MODAL_* set).

    When SANDBOX_BACKEND is runloop/daytona/modal and the optional package
    is not installed, returns None and caller falls back to era/docker.
    """
    global _SANDBOX_BACKEND
    backend_name = (os.getenv("SANDBOX_BACKEND") or "local").strip().lower()
    root = project_root or Path.cwd()

    if backend_name == "local":
        _SANDBOX_BACKEND = LocalSandboxBackend(project_root=root)
        return _SANDBOX_BACKEND

    if backend_name == "runloop":
        try:
            from src.core.sandbox.backends.runloop import RunloopSandboxBackend

            _SANDBOX_BACKEND = RunloopSandboxBackend()
            return _SANDBOX_BACKEND
        except ImportError as e:
            logger.debug("Runloop sandbox not available: %s", e)
            return None

    if backend_name == "daytona":
        try:
            from src.core.sandbox.backends.daytona import DaytonaSandboxBackend

            _SANDBOX_BACKEND = DaytonaSandboxBackend()
            return _SANDBOX_BACKEND
        except ImportError as e:
            logger.debug("Daytona sandbox not available: %s", e)
            return None

    if backend_name == "modal":
        try:
            from src.core.sandbox.backends.modal import ModalSandboxBackend

            _SANDBOX_BACKEND = ModalSandboxBackend()
            return _SANDBOX_BACKEND
        except ImportError as e:
            logger.debug("Modal sandbox not available: %s", e)
            return None

    # docker and era are handled in _execute_code_tool by existing logic
    return None
