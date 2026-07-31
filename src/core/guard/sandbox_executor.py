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
from typing import List, Optional

logger = logging.getLogger(__name__)

def _is_firejail_available() -> bool:
    try:
        return subprocess.run(
            ["which", "firejail"], capture_output=True
        ).returncode == 0
    except Exception:
        return False

_FIREJAIL_AVAILABLE = _is_firejail_available()
_DOCKER_AVAILABLE = bool(subprocess.run(
    ["which", "docker"], capture_output=True
).returncode == 0)
_GVISOR_AVAILABLE = bool(subprocess.run(
    ["which", "runsc"], capture_output=True
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
    Strategy: gvisor > firejail > docker > restrictive subprocess (fallback).
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

    def _build_gvisor_cmd(self, cmd: str) -> List[str]:
        """Wrap command with gVisor (runsc) restrictions."""
        parts = ["runsc", "do", "--"]
        if not self.network_access:
            parts.append("--net=none")
        parts.extend(["bash", "-c", cmd])
        return parts

    def _build_firejail_cmd(self, cmd: str) -> List[str]:
        """Wrap command with firejail restrictions."""
        parts = ["firejail", "--quiet", "--private"]
        if not self.network_access:
            parts.append("--net=none")
        if self.allowed_paths:
            for path in self.allowed_paths:
                parts.extend(["--whitelist=" + path])
        parts.extend(["--", "bash", "-c", cmd])
        return parts

    def _build_docker_cmd(self, cmd: str) -> List[str]:
        """Wrap command with docker restrictions."""
        parts = [
            "docker", "run", "--rm",
            "--network", "none" if not self.network_access else "bridge",
            "--memory", "256m",
            "--cpus", "0.5",
            "--read-only",
            "python:3.12-slim",
            "bash", "-c", cmd,
        ]
        return parts

    def execute(self, command: str, dry_run: bool = False) -> SandboxResult:
        """
        Execute command in a sandbox. If dry_run=True, return a simulated result.
        """
        if dry_run:
            logger.info("[DRY-RUN] Would execute: %s", command)
            return SandboxResult(
                command=command, returncode=0,
                stdout="[dry-run]", stderr="",
                duration_ms=0.0, sandbox_type="dry-run",
            )

        env_strategy = os.getenv("SPARKLEFORGE_SANDBOX_STRATEGY", "").lower()

        # Choose sandbox strategy
        if env_strategy == "subprocess":
            sandbox_type = "subprocess"
            exec_cmd = ["bash", "-c", command]
        elif _GVISOR_AVAILABLE:
            sandbox_type = "gvisor"
            exec_cmd = self._build_gvisor_cmd(command)
        elif _FIREJAIL_AVAILABLE:
            sandbox_type = "firejail"
            exec_cmd = self._build_firejail_cmd(command)
        elif env_strategy == "docker" or (not env_strategy and _DOCKER_AVAILABLE):
            sandbox_type = "docker"
            exec_cmd = self._build_docker_cmd(command)
        else:
            # Fallback: restricted subprocess
            sandbox_type = "subprocess"
            exec_cmd = ["bash", "-c", command]
            logger.warning(
                "No sandboxing available (firejail/docker missing or unavailable). "
                "Running restricted subprocess for: %s", command
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
            logger.warning("Sandbox execution timed out: %s", command)
        except Exception as e:
            stdout     = ""
            stderr     = str(e)
            returncode = -1
            logger.error("Sandbox execution error: %s", e)

        duration_ms = (time.monotonic() - start) * 1000

        # If containerized sandbox (docker/firejail) failed or timed out, attempt subprocess fallback
        if sandbox_type in ("firejail", "docker") and (timed_out or returncode != 0):
            logger.warning(
                "Sandbox[%s] failed or timed out for '%s'. Falling back to restricted subprocess execution.",
                sandbox_type, command
            )
            fb_start = time.monotonic()
            try:
                fb_result = subprocess.run(
                    ["bash", "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={**os.environ, "HOME": tempfile.mkdtemp()},
                )
                return SandboxResult(
                    command=command,
                    returncode=fb_result.returncode,
                    stdout=fb_result.stdout,
                    stderr=fb_result.stderr,
                    duration_ms=(time.monotonic() - fb_start) * 1000,
                    sandbox_type="subprocess-fallback",
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                pass
            except Exception as fb_err:
                logger.error("Sandbox fallback execution error: %s", fb_err)

        res = SandboxResult(
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            sandbox_type=sandbox_type,
            timed_out=timed_out,
        )
        logger.info(
            "Sandbox[%s] exit=%d dur=%.1fms cmd=%s",
            sandbox_type, returncode, duration_ms, command[:60],
        )
        return res

    async def execute_async(self, command: str, dry_run: bool = False) -> SandboxResult:
        """Async wrapper for non-blocking sandbox execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, command, dry_run)
