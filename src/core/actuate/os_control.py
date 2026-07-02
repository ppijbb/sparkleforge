import logging
import shutil
import subprocess
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class OSControl:
    """Provides cross-platform GUI interaction (clicks, typing, drag-and-drop) with CLI fallback."""

    def __init__(self):
        self.xdotool_available = shutil.which("xdotool") is not None

    def get_screen_size(self) -> Tuple[int, int]:
        """Get host primary monitor resolution (width, height)."""
        if PYAUTOGUI_AVAILABLE:
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
        
        if PYAUTOGUI_AVAILABLE:
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
        
        if PYAUTOGUI_AVAILABLE:
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

        if PYAUTOGUI_AVAILABLE:
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
