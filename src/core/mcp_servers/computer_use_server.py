"""Computer Use MCP Server - Virtual desktop screenshot and input control.

Provides a headless X11 virtual display (Xvfb) and exposes atomic tools
for capturing screenshots, controlling the mouse, and sending keyboard input.
Any agent that needs to interact with a GUI application should use these tools.

System prerequisites:
    sudo apt-get install -y xvfb x11-utils scrot
"""

import asyncio
import base64
import io
import json
import logging
import os
import subprocess
from typing import Any

try:
    from fastmcp import FastMCP
    from pydantic import BaseModel, Field

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    BaseModel = None
    Field = None

logger = logging.getLogger(__name__)

# Virtual display configuration (from environment or defaults)
DISPLAY_WIDTH = int(os.getenv("COMPUTER_USE_WIDTH", "1280"))
DISPLAY_HEIGHT = int(os.getenv("COMPUTER_USE_HEIGHT", "800"))
DISPLAY_DEPTH = 24

# Module-level state for the Xvfb display lifecycle
_display = None  # xvfbwrapper.Xvfb instance, started lazily
_pyautogui = None  # imported lazily after display is ready
_display_num: int = 99  # fallback display number

# Initialize FastMCP server
mcp = FastMCP("computer-use")


# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------


class ScreenshotInput(BaseModel):
    """Input for capturing a screenshot."""

    format: str = Field(default="png", description="Image format: png or jpeg")
    quality: int = Field(default=85, description="JPEG quality 1-100", ge=1, le=100)


class MouseClickInput(BaseModel):
    """Input for clicking the mouse."""

    x: int = Field(..., description="X coordinate in pixels")
    y: int = Field(..., description="Y coordinate in pixels")
    button: str = Field(default="left", description="Mouse button: left, right, middle")
    clicks: int = Field(default=1, description="Number of clicks (1=single, 2=double)", ge=1, le=3)
    interval: float = Field(default=0.1, description="Interval between clicks in seconds")


class MouseMoveInput(BaseModel):
    """Input for moving the mouse cursor."""

    x: int = Field(..., description="Target X coordinate in pixels")
    y: int = Field(..., description="Target Y coordinate in pixels")
    duration: float = Field(default=0.25, description="Move duration in seconds")


class MouseScrollInput(BaseModel):
    """Input for scrolling the mouse wheel."""

    x: int = Field(..., description="X coordinate to scroll at")
    y: int = Field(..., description="Y coordinate to scroll at")
    clicks: int = Field(
        default=3,
        description="Number of scroll steps. Positive = scroll up, negative = scroll down",
    )


class TypeTextInput(BaseModel):
    """Input for typing text."""

    text: str = Field(..., description="Text to type", min_length=1)
    interval: float = Field(default=0.05, description="Delay between keystrokes in seconds")


class KeyPressInput(BaseModel):
    """Input for pressing a keyboard key."""

    key: str = Field(
        ...,
        description="Key name (e.g. 'enter', 'escape', 'ctrl+c', 'alt+F4', 'tab')",
        min_length=1,
    )


class OpenAppInput(BaseModel):
    """Input for launching an application."""

    command: str = Field(..., description="Shell command to launch the application", min_length=1)
    wait_seconds: float = Field(
        default=2.0,
        description="Seconds to wait after launching for the app to appear",
        ge=0,
        le=30,
    )


class DisplayInfoInput(BaseModel):
    """Input for getting display information (no parameters needed)."""


# ---------------------------------------------------------------------------
# Display lifecycle helpers
# ---------------------------------------------------------------------------


def _ensure_display() -> int:
    """Start Xvfb if not already running. Returns the display number.

    This is idempotent: multiple tools calling it will reuse the same Xvfb
    instance. The display number is set in DISPLAY environment variable so
    that child processes (browsers, apps) automatically use the virtual display.
    """
    global _display, _pyautogui, _display_num

    if _display is not None:
        return _display_num

    # Try xvfbwrapper first (cleanest lifecycle management)
    try:
        import xvfbwrapper

        vdisplay = xvfbwrapper.Xvfb(
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            colordepth=DISPLAY_DEPTH,
        )
        vdisplay.start()
        _display = vdisplay
        _display_num = vdisplay.vdisplay_num
        logger.info(
            "Started Xvfb virtual display :%d (%dx%d)", _display_num, DISPLAY_WIDTH, DISPLAY_HEIGHT
        )
    except ImportError:
        # xvfbwrapper not installed — try launching Xvfb manually
        logger.warning("xvfbwrapper not available, launching Xvfb subprocess on :%d", _display_num)
        try:
            subprocess.Popen(
                [
                    "Xvfb",
                    f":{_display_num}",
                    "-screen",
                    "0",
                    f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}x{DISPLAY_DEPTH}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import time

            time.sleep(0.5)  # give Xvfb a moment to start
            _display = True  # sentinel: display started via subprocess
        except FileNotFoundError:
            logger.error("Xvfb binary not found. Install with: sudo apt-get install -y xvfb")
            raise RuntimeError("Xvfb not available. Install with: sudo apt-get install -y xvfb")

    # Set DISPLAY so pyautogui / mss / child processes use the virtual display
    os.environ["DISPLAY"] = f":{_display_num}"

    # Import pyautogui now (after DISPLAY is set, X11 connection happens on import)
    try:
        import pyautogui as _pg

        _pg.FAILSAFE = False
        _pg.PAUSE = 0.05
        _pyautogui = _pg
        logger.debug("pyautogui initialized on display :%d", _display_num)
    except Exception as exc:
        logger.warning("pyautogui import failed: %s (mouse/keyboard tools will be limited)", exc)

    return _display_num


def _get_pyautogui():
    """Return the lazily initialized pyautogui module."""
    _ensure_display()
    if _pyautogui is None:
        raise RuntimeError("pyautogui is not available. Install with: pip install pyautogui")
    return _pyautogui


def _capture_screenshot_bytes(fmt: str = "png", quality: int = 85) -> bytes:
    """Capture screen and return raw image bytes."""
    display_num = _ensure_display()

    # Try mss first (fastest)
    try:
        import mss
        import PIL.Image

        with mss.mss() as sct:
            # mss uses DISPLAY env var set by _ensure_display()
            monitor = sct.monitors[0]  # full virtual screen (index 0 = all monitors)
            raw = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            buf = io.BytesIO()
            img.save(buf, format=fmt.upper(), quality=quality if fmt == "jpeg" else None)
            return buf.getvalue()
    except Exception as exc:
        logger.debug("mss screenshot failed (%s), trying scrot fallback", exc)

    # Fallback: scrot subprocess
    try:
        result = subprocess.run(
            ["scrot", "-", "--display", f":{display_num}"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("scrot fallback failed: %s", exc)

    # Last resort: xwd + ImageMagick convert
    try:
        xwd = subprocess.run(
            ["xwd", "-root", "-silent", "-display", f":{display_num}"],
            capture_output=True,
            timeout=10,
        )
        convert = subprocess.run(
            ["convert", "xwd:-", f"{fmt}:-"],
            input=xwd.stdout,
            capture_output=True,
            timeout=10,
        )
        if convert.returncode == 0:
            return convert.stdout
    except Exception as exc:
        logger.debug("xwd/convert fallback failed: %s", exc)

    raise RuntimeError("All screenshot methods failed. Ensure mss or scrot is installed.")


def _success(data: dict[str, Any]) -> str:
    return json.dumps({"success": True, **data})


def _error(message: str) -> str:
    return json.dumps({"success": False, "error": message})


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def screenshot(input: ScreenshotInput) -> str:
    """Capture a screenshot of the virtual display.

    Returns JSON with base64-encoded image data and display metadata.
    The 'data' field contains a base64-encoded PNG/JPEG suitable for
    passing directly to vision-capable LLM APIs.
    """
    try:
        img_bytes = _capture_screenshot_bytes(fmt=input.format, quality=input.quality)
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return _success(
            {
                "display": _display_num,
                "width": DISPLAY_WIDTH,
                "height": DISPLAY_HEIGHT,
                "format": input.format,
                "size_bytes": len(img_bytes),
                "data": encoded,
            }
        )
    except Exception as exc:
        logger.error("screenshot failed: %s", exc)
        return _error(str(exc))


@mcp.tool()
async def mouse_click(input: MouseClickInput) -> str:
    """Click the mouse at the specified screen coordinates.

    Supports single click, double click, and right/middle click.
    Coordinates are in pixels from the top-left corner of the virtual display.
    """
    try:
        pg = _get_pyautogui()
        btn_map = {"left": "left", "right": "right", "middle": "middle"}
        button = btn_map.get(input.button.lower(), "left")
        pg.click(
            x=input.x,
            y=input.y,
            button=button,
            clicks=input.clicks,
            interval=input.interval,
        )
        return _success({"x": input.x, "y": input.y, "button": button, "clicks": input.clicks})
    except Exception as exc:
        logger.error("mouse_click failed: %s", exc)
        return _error(str(exc))


@mcp.tool()
async def mouse_move(input: MouseMoveInput) -> str:
    """Move the mouse cursor to the specified coordinates.

    The move is animated over the specified duration.
    """
    try:
        pg = _get_pyautogui()
        pg.moveTo(input.x, input.y, duration=input.duration)
        return _success({"x": input.x, "y": input.y})
    except Exception as exc:
        logger.error("mouse_move failed: %s", exc)
        return _error(str(exc))


@mcp.tool()
async def mouse_scroll(input: MouseScrollInput) -> str:
    """Scroll the mouse wheel at the specified coordinates.

    Positive clicks values scroll up; negative values scroll down.
    """
    try:
        pg = _get_pyautogui()
        pg.moveTo(input.x, input.y)
        pg.scroll(input.clicks)
        direction = "up" if input.clicks > 0 else "down"
        return _success(
            {"x": input.x, "y": input.y, "clicks": input.clicks, "direction": direction}
        )
    except Exception as exc:
        logger.error("mouse_scroll failed: %s", exc)
        return _error(str(exc))


@mcp.tool()
async def type_text(input: TypeTextInput) -> str:
    """Type text using the keyboard.

    Simulates keyboard input character by character. For special characters
    or key combinations, use key_press instead.
    """
    try:
        pg = _get_pyautogui()
        pg.typewrite(input.text, interval=input.interval)
        return _success({"text_length": len(input.text)})
    except Exception as exc:
        logger.error("type_text failed: %s", exc)
        return _error(str(exc))


@mcp.tool()
async def key_press(input: KeyPressInput) -> str:
    """Press a keyboard key or key combination.

    Supports single keys (e.g. 'enter', 'escape', 'tab', 'space') and
    combinations using '+' separator (e.g. 'ctrl+c', 'ctrl+alt+delete',
    'alt+F4', 'ctrl+shift+t').
    """
    try:
        pg = _get_pyautogui()
        # Handle key combinations (e.g. 'ctrl+c')
        keys = [k.strip() for k in input.key.split("+")]
        if len(keys) > 1:
            pg.hotkey(*keys)
        else:
            pg.press(keys[0])
        return _success({"key": input.key})
    except Exception as exc:
        logger.error("key_press failed: %s", exc)
        return _error(str(exc))


@mcp.tool()
async def open_application(input: OpenAppInput) -> str:
    """Launch an application in the virtual display.

    Starts the application as a background subprocess with DISPLAY set to the
    virtual display, then waits for the specified number of seconds.
    """
    try:
        display_num = _ensure_display()
        env = {**os.environ, "DISPLAY": f":{display_num}"}
        proc = subprocess.Popen(
            input.command,
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.sleep(input.wait_seconds)
        # Check if process is still running (not crashed immediately)
        running = proc.poll() is None
        return _success(
            {
                "command": input.command,
                "pid": proc.pid,
                "running": running,
                "wait_seconds": input.wait_seconds,
            }
        )
    except Exception as exc:
        logger.error("open_application failed: %s", exc)
        return _error(str(exc))


@mcp.tool()
async def get_display_info(input: DisplayInfoInput) -> str:
    """Return information about the current virtual display.

    Reports display number, dimensions, and whether Xvfb is running.
    """
    try:
        display_active = _display is not None
        display_env = os.environ.get("DISPLAY", "not set")
        return _success(
            {
                "display_number": _display_num,
                "display_env": display_env,
                "width": DISPLAY_WIDTH,
                "height": DISPLAY_HEIGHT,
                "color_depth": DISPLAY_DEPTH,
                "xvfb_running": display_active,
                "pyautogui_available": _pyautogui is not None,
            }
        )
    except Exception as exc:
        return _error(str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run():
    """Start the MCP server (called by registry subprocess launcher)."""
    mcp.run(show_banner=False)


def get_mcp():
    return mcp


if __name__ == "__main__":
    run()
