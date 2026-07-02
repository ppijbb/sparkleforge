import asyncio
import logging
import shutil
from typing import List

logger = logging.getLogger(__name__)


class UnifiedPackageManager:
    """Unified wrapper around host package managers (apt, brew, pipx, npm)."""

    def __init__(self):
        pass

    def _get_executable_args(self, manager: str, action: str, package: str) -> List[str]:
        """Resolve subprocess executable list argument structure based on package manager and action."""
        mgr = manager.lower().strip()
        act = action.lower().strip()

        if mgr == "apt":
            if not shutil.which("apt-get"):
                raise ValueError("apt-get executable not found on host system")
            if act == "install":
                return ["apt-get", "install", "-y", package]
            elif act == "uninstall":
                return ["apt-get", "remove", "-y", package]
            elif act == "upgrade":
                # Only upgrade the specific package
                return ["apt-get", "install", "--only-upgrade", "-y", package]

        elif mgr == "brew":
            if not shutil.which("brew"):
                raise ValueError("brew executable not found on host system")
            if act == "install":
                return ["brew", "install", package]
            elif act == "uninstall":
                return ["brew", "uninstall", package]
            elif act == "upgrade":
                return ["brew", "upgrade", package]

        elif mgr == "pipx":
            if not shutil.which("pipx"):
                raise ValueError("pipx executable not found on host system")
            if act == "install":
                return ["pipx", "install", package]
            elif act == "uninstall":
                return ["pipx", "uninstall", package]
            elif act == "upgrade":
                return ["pipx", "upgrade", package]

        elif mgr == "npm":
            if not shutil.which("npm"):
                raise ValueError("npm executable not found on host system")
            if act == "install":
                return ["npm", "install", "-g", package]
            elif act == "uninstall":
                return ["npm", "uninstall", "-g", package]
            elif act == "upgrade":
                return ["npm", "update", "-g", package]

        raise ValueError(f"Unsupported package manager: {manager}")

    async def _execute(self, args: List[str]) -> bool:
        """Run package manager command asynchronously."""
        cmd_str = " ".join(args)
        logger.info(f"UnifiedPackageManager: Executing: {cmd_str}")
        try:
            # We use subprocess exec since arguments are predefined lists (safer than shell)
            proc = await asyncio.create_subprocess_exec(
                args[0],
                *args[1:],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            returncode = proc.returncode

            if returncode == 0:
                logger.info(f"UnifiedPackageManager: Executed successfully")
                return True
            else:
                stderr_str = stderr_bytes.decode(errors="replace")
                logger.error(f"UnifiedPackageManager: Execution failed with returncode {returncode}. Stderr: {stderr_str}")
                return False
        except Exception as e:
            logger.error(f"UnifiedPackageManager: Failed to run command {cmd_str}: {e}")
            return False

    async def install(self, manager: str, package: str) -> bool:
        """Install a package using the specified package manager."""
        try:
            args = self._get_executable_args(manager, "install", package)
            return await self._execute(args)
        except ValueError as e:
            logger.warning(f"UnifiedPackageManager: {e}")
            return False

    async def uninstall(self, manager: str, package: str) -> bool:
        """Uninstall a package using the specified package manager."""
        try:
            args = self._get_executable_args(manager, "uninstall", package)
            return await self._execute(args)
        except ValueError as e:
            logger.warning(f"UnifiedPackageManager: {e}")
            return False

    async def upgrade(self, manager: str, package: str) -> bool:
        """Upgrade a package using the specified package manager."""
        try:
            args = self._get_executable_args(manager, "upgrade", package)
            return await self._execute(args)
        except ValueError as e:
            logger.warning(f"UnifiedPackageManager: {e}")
            return False
