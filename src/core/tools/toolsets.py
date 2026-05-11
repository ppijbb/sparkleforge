import logging
from typing import List

from src.core.tools.registry import ToolCategory, registry

# Register browser tools when importing toolsets
try:
    from src.core.tools.browser_tools import register_browser_tools

    register_browser_tools()
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to register browser tools: {e}")

logger = logging.getLogger(__name__)


class ToolSet:
    """Manager for dynamic tool bundling (Phase 2).

    Allows creating sets of tools for specific tasks like 'research', 'code', etc.
    """

    def __init__(self):
        self.registry = registry

    def get_research_tools(self) -> List[str]:
        """Returns tools relevant for research."""
        return [
            name
            for name, meta in self.registry.tools.items()
            if meta.category in [ToolCategory.SEARCH, ToolCategory.ACADEMIC, ToolCategory.DATA]
        ]

    def get_coding_tools(self) -> List[str]:
        """Returns tools relevant for coding."""
        return [
            name
            for name, meta in self.registry.tools.items()
            if meta.category in [ToolCategory.CODE, ToolCategory.FILE, ToolCategory.GIT]
        ]

    def get_browser_tools(self) -> List[str]:
        """Returns tools relevant for browser automation."""
        return [
            name
            for name, meta in self.registry.tools.items()
            if meta.category == ToolCategory.BROWSER
        ]

    def get_bundled_tools(self, categories: List[ToolCategory]) -> List[str]:
        """Returns tools for specified categories."""
        return [name for name, meta in self.registry.tools.items() if meta.category in categories]


# Global toolset manager
toolsets = ToolSet()
