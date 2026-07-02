import logging
import platform
import subprocess
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class WindowTracker:
    """Tracks active and focus windows on different platform backends (cross-platform)."""

    def __init__(self):
        self.os_type = platform.system().lower()

    def _command_exists(self, cmd: str) -> bool:
        import shutil
        return shutil.which(cmd) is not None

    async def _get_linux_active_window(self) -> Dict[str, Any]:
        """Fetch active window under Linux using xdotool or wmctrl."""
        if self._command_exists("xdotool"):
            try:
                res_id = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                window_id = res_id.stdout.strip()
                res_name = subprocess.run(
                    ["xdotool", "getwindowname", window_id],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                window_name = res_name.stdout.strip()
                res_pid = subprocess.run(
                    ["xdotool", "getwindowpid", window_id],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                pid_str = res_pid.stdout.strip()
                pid = int(pid_str) if res_pid.returncode == 0 and pid_str.isdigit() else None

                return {
                    "window_id": window_id,
                    "title": window_name,
                    "pid": pid,
                    "class": "unknown",
                }
            except Exception as e:
                logger.debug(f"WindowTracker (Linux): xdotool query failed: {e}")

        # wmctrl does not support getting the active window directly without xprop, so we fall back.

        return {"title": "X11 Display Window (fallback)", "pid": None, "window_id": None}

    async def _get_macos_active_window(self) -> Dict[str, Any]:
        """Fetch active window under macOS using osascript."""
        if not self._command_exists("osascript"):
            return {"title": "macOS Window (osascript unavailable)", "pid": None}

        script = """
        tell application "System Events"
            set frontmostProcess to first process whose frontmost is true
            set processName to name of frontmostProcess
            try
                set windowTitle to name of first window of frontmostProcess
            on error
                set windowTitle to "unknown"
            end try
            return processName & "||" & windowTitle
        end tell
        """
        try:
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=True,
            )
            parts = res.stdout.strip().split("||", 1)
            return {
                "application": parts[0] if len(parts) > 0 else "unknown",
                "title": parts[1] if len(parts) > 1 else "unknown",
            }
        except Exception as e:
            logger.debug(f"WindowTracker (macOS): AppleScript failed: {e}")
            return {"title": "macOS Window (fallback)", "pid": None}

    async def get_active_window(self) -> Dict[str, Any]:
        """Get telemetry details of the currently focused/active window."""
        try:
            if self.os_type == "linux":
                return await self._get_linux_active_window()
            elif self.os_type == "darwin":
                return await self._get_macos_active_window()
            else:
                return {
                    "title": f"Active GUI Window on {platform.system()}",
                    "pid": None,
                    "fallback": True,
                }
        except Exception as e:
            logger.error(f"WindowTracker: Failed to fetch active window: {e}")
            return {"title": "error", "error": str(e)}

    async def get_window_list(self) -> List[Dict[str, Any]]:
        """List all visible windows (Linux / X11 fallback)."""
        windows = []
        if self.os_type == "linux" and self._command_exists("wmctrl"):
            try:
                res = subprocess.run(
                    ["wmctrl", "-l"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in res.stdout.splitlines():
                    parts = line.strip().split(None, 3)
                    if len(parts) >= 4:
                        windows.append({
                            "window_id": parts[0],
                            "desktop": parts[1],
                            "host": parts[2],
                            "title": parts[3],
                        })
            except Exception as e:
                logger.debug(f"WindowTracker (Linux): wmctrl -l query failed: {e}")
        else:
            active = await self.get_active_window()
            if active and "title" in active:
                windows.append(active)

        return windows
