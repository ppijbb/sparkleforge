import logging
import subprocess
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PackageInventory:
    """Collects lists of installed packages across multiple package managers."""

    def __init__(self):
        pass

    def _command_exists(self, cmd: str) -> bool:
        """Check if a shell command exists."""
        import shutil
        return shutil.which(cmd) is not None

    async def list_apt_packages(self) -> List[Dict[str, Any]]:
        """List apt packages (Debian/Ubuntu)."""
        if not self._command_exists("dpkg-query"):
            return []

        packages = []
        try:
            import asyncio
            proc = await asyncio.create_subprocess_exec(
                "dpkg-query", "-W", "-f=${Package}\t${Version}\t${Status}\n",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return []
            for line in stdout.decode().splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    packages.append({
                        "name": parts[0],
                        "version": parts[1],
                        "status": parts[2],
                    })
            return packages
        except Exception as e:
            logger.debug(f"PackageInventory: Failed to list apt packages: {e}")
            return []

    async def list_brew_packages(self) -> List[Dict[str, Any]]:
        """List Homebrew packages (macOS/Linux)."""
        if not self._command_exists("brew"):
            return []

        packages = []
        try:
            res = subprocess.run(
                ["brew", "list", "--versions"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in res.stdout.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) >= 2:
                    packages.append({
                        "name": parts[0],
                        "version": parts[1],
                    })
                elif len(parts) == 1:
                    packages.append({
                        "name": parts[0],
                        "version": "unknown",
                    })
            return packages
        except Exception as e:
            logger.debug(f"PackageInventory: Failed to list brew packages: {e}")
            return []

    async def list_pipx_packages(self) -> List[Dict[str, Any]]:
        """List pipx-installed globally accessible CLI tools."""
        if not self._command_exists("pipx"):
            return []

        packages = []
        try:
            res = subprocess.run(
                ["pipx", "list", "--short"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in res.stdout.splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split(None, 1)
                if len(parts) >= 2:
                    packages.append({
                        "name": parts[0],
                        "version": parts[1].strip("(),"),
                    })
            return packages
        except Exception as e:
            logger.debug(f"PackageInventory: Failed to list pipx packages: {e}")
            return []

    async def list_npm_packages(self) -> List[Dict[str, Any]]:
        """List globally installed npm packages."""
        if not self._command_exists("npm"):
            return []

        packages = []
        try:
            res = subprocess.run(
                ["npm", "list", "-g", "--depth=0", "--json"],
                capture_output=True,
                text=True,
            )
            if res.stdout.strip():
                import json
                data = json.loads(res.stdout)
                deps = data.get("dependencies", {})
                for name, info in deps.items():
                    version = "unknown"
                    if isinstance(info, dict):
                        version = info.get("version", "unknown")
                    packages.append({
                        "name": name,
                        "version": version,
                    })
            return packages
        except Exception as e:
            logger.debug(f"PackageInventory: Failed to list npm packages: {e}")
            return []

    async def get_unified_inventory(self) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate all available package inventory states."""
        return {
            "apt": await self.list_apt_packages(),
            "brew": await self.list_brew_packages(),
            "pipx": await self.list_pipx_packages(),
            "npm": await self.list_npm_packages(),
        }
