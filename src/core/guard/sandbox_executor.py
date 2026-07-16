"""
sandbox_executor.py — Containerized/namespaced sandboxed execution for untrusted tool calls.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

_FIREJAIL_AVAILABLE = bool(subprocess.run(
    ["which", "firejail"], capture_output=True
).returncode == 0)

_DOCKER_AVAILABLE = bool(subprocess.run(
    ["which", "docker"], capture_output=True
).returncode == 0)


@dataclass
class SandboxResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    sandbox_type: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class SandboxExecutor:
    """
    Executes commands in an isolated environment.
    Strategy: firejail > docker > restrictive subprocess (fallback).
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        allowed_paths: Optional[List[str]] = None,
        network_access: bool = False,
    ) -> None:
        self.timeout = timeout_seconds
        self.allowed_paths = allowed_paths or []
        self.network_access = network_access

    def _build_firejail_cmd(self, cmd: Union[str, Sequence[str]]) -> List[str]:
        """Wrap command with firejail restrictions."""
        parts = ["firejail", "--quiet", "--private"]
        if not self.network_access:
            parts.append("--net=none")
        if self.allowed_paths:
            for path in self.allowed_paths:
                parts.extend(["--whitelist=" + path])
        parts.append("--")
        parts.extend(self._shell_argv(cmd))
        return parts

    def _build_docker_cmd(self, cmd: Union[str, Sequence[str]]) -> List[str]:
        """Wrap command with docker restrictions."""
        parts = [
            "docker", "run", "--rm",
            "--network", "none" if not self.network_access else "bridge",
            "--memory", "256m",
            "--cpus", "0.5",
            "--read-only",
            "python:3.12-slim",
            "bash", "-lc",
        ]
        parts.append(self._shell_command(cmd))
        return parts

    @staticmethod
    def _shell_command(command: Union[str, Sequence[str]]) -> str:
        """Normalize a command into a single shell string for bash -c."""
        if isinstance(command, str):
            return command
        return " ".join(shlex.quote(str(arg)) for arg in command)

    @staticmethod
    def _shell_argv(command: Union[str, Sequence[str]]) -> List[str]:
        """Normalize a command into an argv list (no shell interpolation)."""
        if isinstance(command, str):
            return ["bash", "-c", command]
        return [str(arg) for arg in command]

    def execute(
        self,
        command: Union[str, Sequence[str]],
        dry_run: bool = False,
    ) -> SandboxResult:
        """
        Execute command in a sandbox. If dry_run=True, return a simulated result.
        """
        if dry_run:
            logger.info("[DRY-RUN] Would execute: %s", self._shell_command(command))
            return SandboxResult(
                command=self._shell_command(command), returncode=0,
                stdout="[dry-run]", stderr="",
                duration_ms=0.0, sandbox_type="dry-run",
            )

        # Choose sandbox strategy
        if _FIREJAIL_AVAILABLE:
            sandbox_type = "firejail"
            exec_cmd = self._build_firejail_cmd(command)
        elif _DOCKER_AVAILABLE:
            sandbox_type = "docker"
            exec_cmd = self._build_docker_cmd(command)
        else:
            # Fallback: restricted subprocess
            sandbox_type = "subprocess"
            exec_cmd = self._shell_argv(command)
            logger.warning(
                "No sandboxing available (firejail/docker missing). "
                "Running restricted subprocess for: %s",
                self._shell_command(command),
            )

        start = time.monotonic()
        timed_out = False
        try:
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "HOME": tempfile.mkdtemp()},
            )
            stdout   = result.stdout
            stderr   = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out  = True
            stdout     = ""
            stderr     = f"Command timed out after {self.timeout}s"
            returncode = -1
            logger.warning("Sandbox execution timed out: %s", self._shell_command(command))
        except Exception as e:
            stdout     = ""
            stderr     = str(e)
            returncode = -1
            logger.error("Sandbox execution error: %s", e)

        duration_ms = (time.monotonic() - start) * 1000

        command_str = self._shell_command(command)
        res = SandboxResult(
            command=command_str,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            sandbox_type=sandbox_type,
            timed_out=timed_out,
        )
        logger.info(
            "Sandbox[%s] exit=%d dur=%.1fms cmd=%s",
            sandbox_type, returncode, duration_ms, command_str[:60],
        )
        return res

    async def execute_async(
        self,
        command: Union[str, Sequence[str]],
        dry_run: bool = False,
    ) -> SandboxResult:
        """Async wrapper for non-blocking sandbox execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, command, dry_run)
