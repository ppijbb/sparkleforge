"""Skills provider MCP server - exposes get_skill_instructions for progressive disclosure.

When USE_PROGRESSIVE_SKILL_DISCLOSURE is enabled, agents receive skill summaries only
and can call this tool to load full instructions on demand.
"""

import logging
from typing import Any

try:
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None

logger = logging.getLogger(__name__)

mcp = FastMCP("sparkleforge-skills") if FASTMCP_AVAILABLE else None


if FASTMCP_AVAILABLE and mcp is not None:

    @mcp.tool()
    async def get_skill_instructions(skill_id: str) -> dict[str, Any]:
        """Load full instructions, overview, and usage for a skill by ID.

        Use this when you need detailed guidance for a skill that was only summarized
        in the system message. Returns overview, instructions, usage, and examples.
        """
        try:
            from src.core.skills_manager import get_skill_manager

            sm = get_skill_manager()
            skill = sm.load_skill(skill_id)
            if not skill:
                return {
                    "success": False,
                    "error": f"Skill not found: {skill_id}",
                    "overview": "",
                    "instructions": "",
                    "usage": "",
                    "examples": "",
                }
            return {
                "success": True,
                "skill_id": skill_id,
                "name": skill.metadata.name,
                "overview": getattr(skill, "overview", "") or "",
                "instructions": getattr(skill, "instructions", "") or "",
                "usage": getattr(skill, "usage", "") or "",
                "examples": getattr(skill, "examples", "") or "",
            }
        except Exception as e:
            logger.warning("get_skill_instructions failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "overview": "",
                "instructions": "",
                "usage": "",
                "examples": "",
            }


def main() -> None:
    if mcp is None:
        raise RuntimeError("FastMCP not available")
    mcp.run()


if __name__ == "__main__":
    main()
