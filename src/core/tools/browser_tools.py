import logging
from typing import Any, Dict

from src.core.tools.registry import ToolCategory, ToolMetadata, registry

logger = logging.getLogger(__name__)


async def cdp_navigate(url: str) -> Dict[str, Any]:
    """Navigate to a URL using the CDP browser."""
    from src.automation.cdp_browser_controller import get_cdp_controller

    ctrl = get_cdp_controller()
    state = await ctrl.navigate(url)
    return {"success": True, "url": state.url, "title": state.title}


async def cdp_click(x: int = 0, y: int = 0, selector: str = "") -> Dict[str, Any]:
    """Click at coordinates or via selector using CDP."""
    from src.automation.cdp_browser_controller import get_cdp_controller

    ctrl = get_cdp_controller()
    action = {"action": "click", "x": x, "y": y}
    if selector:
        action["selector"] = selector
    results = await ctrl.interact([action])
    res = results[0]
    return {"success": res.success, "data": res.data, "error": res.error}


async def cdp_type_text(text: str) -> Dict[str, Any]:
    """Type text using CDP."""
    from src.automation.cdp_browser_controller import get_cdp_controller

    ctrl = get_cdp_controller()
    results = await ctrl.interact([{"action": "type", "value": text}])
    res = results[0]
    return {"success": res.success, "data": res.data, "error": res.error}


async def cdp_screenshot(filename: str = None, full_page: bool = False) -> Dict[str, Any]:
    """Take a screenshot using CDP."""
    from src.automation.cdp_browser_controller import get_cdp_controller

    ctrl = get_cdp_controller()
    action = {"action": "screenshot"}
    if filename:
        action["filename"] = filename
    if full_page:
        action["full_page"] = full_page
    results = await ctrl.interact([action])
    res = results[0]
    return {"success": res.success, "data": res.data, "error": res.error}


async def cdp_js(script: str) -> Dict[str, Any]:
    """Execute JavaScript via CDP."""
    from src.automation.cdp_browser_controller import get_cdp_controller

    ctrl = get_cdp_controller()
    results = await ctrl.interact([{"action": "execute_js", "script": script}])
    res = results[0]
    return {"success": res.success, "data": res.data, "error": res.error}


async def cdp_extract_text() -> Dict[str, Any]:
    """Extract full page text/markdown."""
    from src.automation.cdp_browser_controller import get_cdp_controller

    ctrl = get_cdp_controller()
    extracted = await ctrl.extract({"full_text": True, "metadata": True})
    return {"success": True, "data": extracted}


async def cdp_page_info() -> Dict[str, Any]:
    """Get current page state."""
    from src.automation.cdp_browser_controller import get_cdp_controller

    ctrl = get_cdp_controller()
    state = await ctrl.get_page_state()
    return {
        "success": True,
        "url": state.url,
        "title": state.title,
        "length": len(state.content_markdown),
    }


def register_browser_tools():
    """Register all CDP browser tools in the global registry."""
    logger.info("Registering CDP browser tools")

    registry.register(
        ToolMetadata(
            name="cdp_navigate",
            description="Navigate to a URL using the CDP browser and wait for load. Use this for the first navigation.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to navigate to"}},
                "required": ["url"],
            },
            category=ToolCategory.BROWSER,
        ),
        cdp_navigate,
    )

    registry.register(
        ToolMetadata(
            name="cdp_click",
            description="Click at specific (x, y) coordinates or using a CSS selector. Coordinates are preferred to bypass framework intercepts.",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "selector": {
                        "type": "string",
                        "description": "CSS selector fallback if coordinates are unknown",
                    },
                },
            },
            category=ToolCategory.BROWSER,
        ),
        cdp_click,
    )

    registry.register(
        ToolMetadata(
            name="cdp_type_text",
            description="Type text exactly as provided. Make sure to focus an input element first.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The text to type"}},
                "required": ["text"],
            },
            category=ToolCategory.BROWSER,
        ),
        cdp_type_text,
    )

    registry.register(
        ToolMetadata(
            name="cdp_screenshot",
            description="Capture a screenshot. Essential for deciding where to click next.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Optional file path to save"},
                    "full_page": {
                        "type": "boolean",
                        "description": "Whether to capture beyond viewport (default False)",
                    },
                },
            },
            category=ToolCategory.BROWSER,
        ),
        cdp_screenshot,
    )

    registry.register(
        ToolMetadata(
            name="cdp_js",
            description="Execute JavaScript in the page context. Return values must be JSON serializable.",
            parameters={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "The JavaScript code to execute"}
                },
                "required": ["script"],
            },
            category=ToolCategory.BROWSER,
        ),
        cdp_js,
    )

    registry.register(
        ToolMetadata(
            name="cdp_extract_text",
            description="Extract the full text/markdown of the current page.",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.BROWSER,
        ),
        cdp_extract_text,
    )

    registry.register(
        ToolMetadata(
            name="cdp_page_info",
            description="Get info about the current page state (URL, title).",
            parameters={"type": "object", "properties": {}},
            category=ToolCategory.BROWSER,
        ),
        cdp_page_info,
    )
