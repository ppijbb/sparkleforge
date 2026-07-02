import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Basic safety blacklist for commands
COMMAND_BLACKLIST = [
    "rm -rf /",
    "rm -rf /*",
    "chmod -R 777",
    "chmod 777 -R",
    ":(){ :|:& };:",  # Fork bomb
    "mkfs",
    "dd if=/dev/",
]

class SecureShellExecutor:
    """Executes host shell commands asynchronously with strict timeout, dry-run, and security filters."""

    def __init__(self, blacklist: List[str] = None):
        self.blacklist = blacklist if blacklist is not None else COMMAND_BLACKLIST

    def _is_safe(self, command: str) -> bool:
        """Check if command contains blacklisted items."""
        cmd_lower = command.strip().lower()
        for pattern in self.blacklist:
            if pattern.lower() in cmd_lower:
                return False
        return True

    async def run_command(
        self, 
        command: str, 
        timeout: float = 30.0, 
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Asynchronously run a shell command on the host."""
        if not self._is_safe(command):
            logger.warning(f"SecureShellExecutor: Command blocked by policy: {command}")
            return {
                "stdout": "",
                "stderr": "Command blocked by security policy.",
                "returncode": -1,
                "status": "blocked"
            }

        if dry_run:
            logger.info(f"SecureShellExecutor [Dry-Run]: {command}")
            return {
                "stdout": f"[dry-run] executed: {command}",
                "stderr": "",
                "returncode": 0,
                "status": "success"
            }

        try:
            # Spawn asynchronous shell process
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            # Await with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=timeout
                )
                stdout = stdout_bytes.decode(errors="replace")
                stderr = stderr_bytes.decode(errors="replace")
                returncode = proc.returncode
                
                status = "success" if returncode == 0 else "failed"
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode,
                    "status": status
                }
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                logger.error(f"SecureShellExecutor: Command timed out after {timeout}s: {command}")
                return {
                    "stdout": "",
                    "stderr": f"Command execution timed out after {timeout} seconds.",
                    "returncode": -1,
                    "status": "timeout"
                }
        except Exception as e:
            logger.error(f"SecureShellExecutor: Command execution error: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "status": "failed"
            }
