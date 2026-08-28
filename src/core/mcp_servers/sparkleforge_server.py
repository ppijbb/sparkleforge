"""SparkleForge MCP Server (Anvil Phase B-1).

Exposes SparkleForge itself as an MCP server, so an external MCP client
(e.g. a separate Claude Code session, via the Claude Code plugin manifest at
``.claude-plugin/plugin.json``) can run a SparkleForge request as a tool call
instead of shelling out to the ``sparkleforge``/``sparkle`` CLI.

Thin wrapper around ``src.sdk.run()`` -- the same in-process entrypoint the
CLI's headless ``--prompt`` mode and other embedders already use.
"""

import json
import logging

try:
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None

logger = logging.getLogger(__name__)

mcp = FastMCP("sparkleforge") if FASTMCP_AVAILABLE else None


if mcp is not None:

    @mcp.tool()
    async def run_task(prompt: str) -> str:
        """Run a SparkleForge research/coworker request and return its result.

        Runs the same code path as ``sparkleforge run "<prompt>"``, in-process
        (see ``src/sdk.py``). Returns a JSON string; on failure, a JSON object
        with ``success: false`` and an ``error`` field instead of raising, so
        MCP clients always get a structured response.
        """
        from src.sdk import run

        try:
            result = await run(prompt)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error("[SparkleForgeMCPServer] run_task failed: %s", e, exc_info=True)
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def run():
    """Run the SparkleForge MCP server (stdio transport)."""
    if mcp is None:
        raise RuntimeError(
            "fastmcp is not installed; cannot run the SparkleForge MCP server. "
            "Install with: pip install fastmcp"
        )
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run()
