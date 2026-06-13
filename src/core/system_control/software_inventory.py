"""Read-only software inventory collection."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Any


PACKAGE_MANAGERS = ["apt", "dnf", "yum", "pacman", "brew", "snap", "flatpak"]
BROWSERS = ["google-chrome", "chromium", "firefox", "brave-browser", "microsoft-edge"]
EDITORS = ["code", "cursor", "vim", "nvim", "emacs", "nano"]


def _run(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or result.stderr.strip()


def _version(command: str, *args: str) -> str | None:
    if not shutil.which(command):
        return None
    output = _run([command, *args])
    return output.splitlines()[0] if output else None


def detect_package_managers() -> dict[str, str]:
    """Return installed package managers and their executable paths."""
    return {name: path for name in PACKAGE_MANAGERS if (path := shutil.which(name))}


def collect_software_inventory() -> dict[str, Any]:
    """Collect a lightweight read-only software inventory."""
    package_managers = detect_package_managers()
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "package_managers": package_managers,
        "python": {
            "executable": sys.executable,
            "version": sys.version.splitlines()[0],
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "node": {
            "node": _version("node", "--version"),
            "npm": _version("npm", "--version"),
            "global_packages": _run(["npm", "list", "-g", "--depth=0"], timeout=5.0)
            if shutil.which("npm")
            else None,
        },
        "docker": {
            "version": _version("docker", "--version"),
            "containers": _run(["docker", "ps", "--format", "{{json .}}"], timeout=5.0)
            if shutil.which("docker")
            else None,
        },
        "services": {
            "systemctl_available": shutil.which("systemctl") is not None,
            "running_sample": _run(
                ["systemctl", "list-units", "--type=service", "--state=running", "--no-pager"],
                timeout=5.0,
            )
            if shutil.which("systemctl")
            else None,
        },
        "browsers": {
            name: _version(name, "--version") for name in BROWSERS if shutil.which(name)
        },
        "editors": {name: _version(name, "--version") for name in EDITORS if shutil.which(name)},
    }
