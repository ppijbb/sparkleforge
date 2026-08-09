import logging
import shutil
import subprocess
from typing import Tuple

logger = logging.getLogger(__name__)



class OSControl:
    """Provides cross-platform GUI interaction (clicks, typing, drag-and-drop) with CLI fallback."""

    def __init__(self):
        self.xdotool_available = shutil.which("xdotool") is not None
        self._pyautogui = None
        self._pyautogui_checked = False

    def _get_pyautogui(self):
        """Lazily import pyautogui only when GUI automation is actually needed.

        Importing pyautogui at module load time emits Xlib/SyntaxWarning noise
        to stderr on every SparkleForge invocation, even for sessions that never
        use computer-use features. Defer the import to the first GUI call.
        """
        if self._pyautogui_checked:
            return self._pyautogui
        self._pyautogui_checked = True
        try:
            import pyautogui
            self._pyautogui = pyautogui
        except Exception:
            # pyautogui may raise KeyError('DISPLAY') or X11-related
            # exceptions in headless environments (e.g. GitHub Actions) where
            # no X server is available. Fall back gracefully instead of
            # crashing the bootstrap.
            self._pyautogui = None
        return self._pyautogui

    def get_screen_size(self) -> Tuple[int, int]:
        """Get host primary monitor resolution (width, height)."""
        pyautogui = self._get_pyautogui()
        if pyautogui is not None:
            try:
                w, h = pyautogui.size()
                return int(w), int(h)
            except Exception:
                pass

        if self.xdotool_available:
            try:
                res = subprocess.run(
                    ["xdotool", "getdisplaygeometry"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                parts = res.stdout.strip().split()
                if len(parts) >= 2:
                    return int(parts[0]), int(parts[1])
            except Exception:
                pass

        # Fallback default
        return 1920, 1080

    async def click(self, x: int, y: int) -> bool:
        """Move cursor to coordinate (x, y) and click."""
        logger.info(f"OSControl: Click at ({x}, {y}) requested")
        
        pyautogui = self._get_pyautogui()
        if pyautogui is not None:
            try:
                pyautogui.click(x, y)
                return True
            except Exception as e:
                logger.debug(f"OSControl (pyautogui): click failed: {e}")

        if self.xdotool_available:
            try:
                # Use subprocess.run asynchronously or sync fallback
                res = subprocess.run(
                    ["xdotool", "mousemove", str(x), str(y), "click", "1"],
                    capture_output=True,
                    check=True,
                )
                return res.returncode == 0
            except Exception as e:
                logger.debug(f"OSControl (xdotool): click failed: {e}")

        logger.warning(f"OSControl: Click at ({x}, {y}) mocked (no GUI automation libraries available)")
        return True

    async def type_text(self, text: str) -> bool:
        """Type a string at the current focus location."""
        logger.info(f"OSControl: Type text: '{text[:20]}...' requested")
        
        pyautogui = self._get_pyautogui()
        if pyautogui is not None:
            try:
                pyautogui.write(text)
                return True
            except Exception as e:
                logger.debug(f"OSControl (pyautogui): write failed: {e}")

        if self.xdotool_available:
            try:
                res = subprocess.run(
                    ["xdotool", "type", text],
                    capture_output=True,
                    check=True,
                )
                return res.returncode == 0
            except Exception as e:
                logger.debug(f"OSControl (xdotool): write failed: {e}")

        logger.warning(f"OSControl: Typing mocked (no GUI automation libraries available)")
        return True

    async def drag_to(self, x: int, y: int) -> bool:
        """Drag from current position to (x, y)."""
        logger.info(f"OSControl: Drag to ({x}, {y}) requested")

        pyautogui = self._get_pyautogui()
        if pyautogui is not None:
            try:
                pyautogui.dragTo(x, y)
                return True
            except Exception as e:
                logger.debug(f"OSControl (pyautogui): drag failed: {e}")

        if self.xdotool_available:
            try:
                res = subprocess.run(
                    ["xdotool", "drag", str(x), str(y)],
                    capture_output=True,
                    check=True,
                )
                return res.returncode == 0
            except Exception as e:
                logger.debug(f"OSControl (xdotool): drag failed: {e}")

        logger.warning(f"OSControl: Drag to ({x}, {y}) mocked (no GUI automation libraries available)")
        return True
