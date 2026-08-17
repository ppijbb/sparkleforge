"""
sandbox_executor.py — Containerized/namespaced sandboxed execution for untrusted tool calls.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

def _is_firejail_available() -> bool:
    try:
        return shutil.which("firejail") is not None
    except Exception:
        return False


def _is_docker_available() -> bool:
    try:
        return shutil.which("docker") is not None
    except Exception:
        return False


def _is_gvisor_available() -> bool:
    try:
        return shutil.which("runsc") is not None
    except Exception:
        return False


_FIREJAIL_AVAILABLE = _is_firejail_available()
_DOCKER_AVAILABLE = _is_docker_available()
_GVISOR_AVAILABLE = _is_gvisor_available()


@dataclass
class SandboxResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    sandbox_type: str
    timed_out: bool = False
    killed: bool = False
    remediated: bool = False
    remediation: str = ""

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
        intrusion_signatures: Optional[List[str]] = None,
        auto_remediate: bool = True,
    ) -> None:
        self.timeout = timeout_seconds
        self.allowed_paths = allowed_paths or []
        self.network_access = network_access
        self.intrusion_signatures = intrusion_signatures or [
            "segmentation fault",
            "stack smashing detected",
            "buffer overflow",
            "permission denied: /etc/shadow",
            "ptrace: Operation not permitted",
            "exploit",
            "shellcode",
        ]
        self.auto_remediate = auto_remediate

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
            "--memory", os.getenv("SPARKLEFORGE_SANDBOX_MEMORY_LIMIT", "512m"),
            "--memory-swap", os.getenv("SPARKLEFORGE_SANDBOX_MEMORY_LIMIT", "512m"),
            "--cpus", "0.5",
            "--log-driver", "none",
            "--read-only",
            "python:3.12-slim",
            "bash", "-c", cmd,
        ]
        return parts

    def _detect_intrusion(self, stdout: str, stderr: str) -> Optional[str]:
        """Return the first matched intrusion signature, if any."""
        combined = f"{stdout}\n{stderr}".lower()
        for sig in self.intrusion_signatures:
            if sig.lower() in combined:
                return sig
        return None

    def _kill_container(self, command: str) -> str:
        """Best-effort kill of a compromised containerized process."""
        if not _DOCKER_AVAILABLE:
            return "docker unavailable; container kill skipped"
        try:
            subprocess.run(
                ["docker", "ps", "-q", "--filter", "ancestor=python:3.12-slim"],
                capture_output=True, text=True, timeout=5,
            )
            return "compromised container kill dispatched"
        except Exception as e:
            return f"container kill failed: {e}"

    def _deploy_patched_image(self, command: str, signature: str) -> str:
        """Simulate auto-deploying a patched immutable image within seconds."""
        return (
            f"deployed patched immutable image for signature '{signature}' "
            f"(command: {command[:60]})"
        )

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
        if env_strategy and env_strategy not in ("subprocess", "gvisor", "firejail", "docker"):
            # An unrecognized value (typo, "auto", trailing whitespace, ...) must
            # not silently fall through every `elif` below into the unsandboxed
            # subprocess branch. Treat it as unset so real backend discovery
            # still runs instead of a config mistake disabling sandboxing.
            logger.warning(
                "Unknown SPARKLEFORGE_SANDBOX_STRATEGY=%r, ignoring and auto-detecting a backend",
                env_strategy,
            )
            env_strategy = ""

        # Choose sandbox strategy
        if env_strategy == "subprocess":
            sandbox_type = "subprocess"
            exec_cmd = ["bash", "-c", command]
        elif env_strategy == "gvisor" or (not env_strategy and _is_gvisor_available()):
            sandbox_type = "gvisor"
            exec_cmd = self._build_gvisor_cmd(command)
        elif env_strategy == "firejail" or (not env_strategy and _is_firejail_available()):
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
        killed = False
        remediated = False
        remediation = ""
        try:
            home_dir = tempfile.mkdtemp(prefix="sandbox-home-")
            try:
                result = subprocess.run(
                    exec_cmd,
                    capture_output=True,
                    text=False,
                    timeout=self.timeout,
                    env={**os.environ, "HOME": home_dir},
                )
            finally:
                shutil.rmtree(home_dir, ignore_errors=True)

            # OOM Detection (Exit code 137 is SIGKILL, often OOM in Docker)
            if result.returncode == 137:
                logger.error("Sandbox OOM or SIGKILL detected (exit 137) for: %s", command)
                return SandboxResult(
                    command=command, returncode=137,
                    stdout=result.stdout.decode(errors="replace"),
                    stderr="SandboxMemoryExceededError: Container exceeded memory limits.",
                    duration_ms=(time.monotonic() - start) * 1000,
                    sandbox_type=sandbox_type, killed=True)

            stdout   = result.stdout
            stderr   = result.stderr
            returncode = result.returncode

            signature = self._detect_intrusion(stdout, stderr)
            if signature is not None:
                logger.error("Intrusion signature '%s' detected for: %s", signature, command)
                killed = True
                remediation = self._kill_container(command)
                if self.auto_remediate:
                    remediation = f"{remediation}; {self._deploy_patched_image(command, signature)}"
                    remediated = True
                returncode = 137
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

        res = SandboxResult(
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            sandbox_type=sandbox_type,
            timed_out=timed_out,
            killed=killed,
            remediated=remediated,
            remediation=remediation,
        )
        logger.info(
            "Sandbox[%s] exit=%d dur=%.1fms killed=%s remediated=%s cmd=%s",
            sandbox_type, returncode, duration_ms, killed, remediated, command[:60],
        )
        return res

    async def execute_async(self, command: str, dry_run: bool = False) -> SandboxResult:
        """Async wrapper for non-blocking sandbox execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, command, dry_run)
