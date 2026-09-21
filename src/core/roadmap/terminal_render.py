"""Render a pty-captured terminal transcript to an actual image.

Feeding an LLM the raw ANSI text (as `cli_ux_audit.build_cli_ux_audit_prompt`
does) misses rendering-only bugs: a missing font glyph shows up as a blank
box on screen but as a perfectly normal-looking character in text. Rendering
to pixels and judging via a vision model is the only way to catch that class
of bug -- confirmed by hand (a pyte+PIL render caught a real one: every emoji
status glyph rendered as tofu because this environment has no emoji font).
"""

from __future__ import annotations

import base64
import io

import pyte
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 100, 40
CELL_W, CELL_H = 9, 18

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)

_ANSI_RGB: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20), "red": (205, 49, 49), "green": (13, 188, 121),
    "yellow": (229, 229, 16), "blue": (36, 114, 200), "magenta": (188, 63, 188),
    "cyan": (17, 168, 205), "white": (229, 229, 229),
    "brightblack": (102, 102, 102), "brightred": (241, 76, 76),
    "brightgreen": (35, 209, 139), "brightyellow": (245, 245, 67),
    "brightblue": (59, 142, 234), "brightmagenta": (214, 112, 214),
    "brightcyan": (41, 184, 219), "brightwhite": (255, 255, 255),
    "default": (204, 204, 204),
}


def _load_font() -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, 14)
        except OSError:
            continue
    return ImageFont.load_default()


def render_transcript_to_png(text: str) -> bytes:
    """Interpret `text` (raw ANSI, e.g. from BaseCLIAgent._execute_command_pty)
    as a vt100 stream and rasterize the resulting screen buffer to PNG bytes."""
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.Stream(screen)
    stream.feed(text)

    img = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    font = _load_font()

    used_rows = 0
    for y, line in screen.buffer.items():
        if y >= ROWS:
            continue
        for x, char in line.items():
            if x >= COLS:
                continue
            if char.data and char.data.strip():
                used_rows = max(used_rows, y)
            fg = _ANSI_RGB.get(char.fg, _ANSI_RGB["default"])
            bg = _ANSI_RGB.get(char.bg) if char.bg != "default" else None
            px, py = x * CELL_W, y * CELL_H
            if bg:
                draw.rectangle([px, py, px + CELL_W, py + CELL_H], fill=bg)
            if char.data and char.data != " ":
                draw.text((px, py), char.data, font=font, fill=fg)

    img = img.crop((0, 0, COLS * CELL_W, min(used_rows + 2, ROWS) * CELL_H))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
