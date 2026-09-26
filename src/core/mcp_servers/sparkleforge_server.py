"""SparkleForge MCP Server (Anvil Phase B-1, extended #1551).

Exposes SparkleForge itself as an MCP server, so an external MCP client
(e.g. a separate Claude Code session, Codex, Claude Desktop -- see the
top-level ``.mcp.json`` or the Claude Code plugin manifest at
``.claude-plugin/plugin.json``) can drive SparkleForge's distinctive
capabilities as tool calls instead of shelling out to the
``sparkleforge``/``sparkle`` CLI.

Tools:
- ``run_task``: blocking one-shot -- runs a request and waits for the result.
  Thin wrapper around ``src.sdk.run()``.
- ``start_research`` / ``get_report``: non-blocking job model for anything
  that might take longer than an MCP client's request timeout. Thin wrapper
  around ``src.sdk.submit_job()`` / ``get_job_status()`` / ``get_report()``
  (the same durable-to-restart job registry ``src/web/status_api.py`` uses,
  see #1654).
- ``nightwelding_status``: lists tracked Nightwelding auto-fix items
  (``src/core/nightwelding/models.py``'s local queue).
- ``search_skills`` / ``get_skill``: query the skill marketplace's local
  share directory (``src/core/anvil/skill_marketplace.py``).

Every tool returns a JSON string and never raises: a failure comes back as
``{"success": false, "error": "..."}`` (or a tool-specific not-found shape),
so MCP clients always get a structured response.
"""

import json
import logging
import os

try:
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None

logger = logging.getLogger(__name__)

mcp = FastMCP("sparkleforge") if FASTMCP_AVAILABLE else None

_DEFAULT_SKILL_SHARE_DIR = os.path.expanduser("~/.sparkleforge/skill_marketplace")


def _skill_share_backend():
    from src.core.anvil.skill_marketplace import LocalSkillShareBackend

    share_dir = os.environ.get("SPARKLEFORGE_SKILL_MARKETPLACE_DIR", _DEFAULT_SKILL_SHARE_DIR)
    return LocalSkillShareBackend(share_dir)


if mcp is not None:

    @mcp.tool()
    async def run_task(prompt: str) -> str:
        """Run a SparkleForge research/coworker request and return its result.

        Runs the same code path as ``sparkleforge run "<prompt>"``, in-process
        (see ``src/sdk.py``). Blocks until finished -- for anything that might
        run long, prefer ``start_research``/``get_report`` instead. Returns a
        JSON string; on failure, a JSON object with ``success: false`` and an
        ``error`` field instead of raising, so MCP clients always get a
        structured response.
        """
        from src.sdk import run

        try:
            result = await run(prompt)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error("[SparkleForgeMCPServer] run_task failed: %s", e, exc_info=True)
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

    @mcp.tool()
    async def start_research(query: str) -> str:
        """Submit a SparkleForge research/coworker request and return immediately.

        Returns ``{"job_id": "..."}``; poll ``get_report(job_id)`` for status
        and the eventual result. Use this instead of ``run_task`` for anything
        that might exceed an MCP client's request timeout.
        """
        from src.sdk import submit_job

        try:
            job_id = await submit_job(query)
            return json.dumps({"job_id": job_id}, ensure_ascii=False)
        except Exception as e:
            logger.error("[SparkleForgeMCPServer] start_research failed: %s", e, exc_info=True)
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

    @mcp.tool()
    async def get_report(job_id: str) -> str:
        """Get the status of (and, once finished, the report from) a job
        started with ``start_research``.

        Always returns ``{"job_id", "status"}``; adds ``"result"`` once
        ``status`` is ``"completed"``, or ``"error"`` if ``"failed"``. A
        ``job_id`` that was never submitted here returns
        ``{"success": false, "error": "..."}``.
        """
        from src.sdk import get_job_status, get_report as sdk_get_report

        try:
            status = await get_job_status(job_id)
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
        except Exception as e:
            logger.error("[SparkleForgeMCPServer] get_report failed: %s", e, exc_info=True)
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

        payload = {"job_id": job_id, "status": status.status}
        if status.status == "completed":
            payload["result"] = await sdk_get_report(job_id)
        elif status.status == "failed":
            payload["error"] = status.error
        return json.dumps(payload, ensure_ascii=False, default=str)

    @mcp.tool()
    async def nightwelding_status() -> str:
        """List Nightwelding auto-fix items this machine has tracked.

        Each item: issue_number, status (queued/writing_test/red/implementing/
        green/draft_opened/failed), branch, pr_url, failure_reason, updated_at.
        Local to this machine's ``~/.sparkleforge/nightwelding/queue.json`` --
        not a live query against GitHub.
        """
        try:
            from src.core.nightwelding.models import NightweldingQueue

            items = NightweldingQueue().list()
            return json.dumps([item.to_dict() for item in items], ensure_ascii=False)
        except Exception as e:
            logger.error(
                "[SparkleForgeMCPServer] nightwelding_status failed: %s", e, exc_info=True
            )
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

    @mcp.tool()
    async def search_skills(query: str) -> str:
        """Search the skill marketplace's local share directory.

        Returns a JSON list of ``{"name", "description", "score"}``, best
        match first. Empty list if nothing has been published there yet.
        """
        try:
            results = _skill_share_backend().search(query)
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            logger.error("[SparkleForgeMCPServer] search_skills failed: %s", e, exc_info=True)
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

    @mcp.tool()
    async def get_skill(skill_id: str) -> str:
        """Get one published skill's manifest (name, description, code, metadata)
        by name from the skill marketplace's local share directory.

        Returns ``{"success": false, "error": "..."}`` if no skill with that
        name has been published there.
        """
        try:
            manifest = _skill_share_backend().read_manifest(skill_id)
            if manifest is None:
                return json.dumps(
                    {"success": False, "error": f"skill '{skill_id}' not found"},
                    ensure_ascii=False,
                )
            # to_dict() deliberately omits raw code (only its sha256) -- the
            # on-disk bundle format (LocalSkillShareBackend.publish()) adds it
            # back explicitly; mirror that here so get_skill returns the code.
            payload = manifest.to_dict()
            payload["code"] = manifest.code
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error("[SparkleForgeMCPServer] get_skill failed: %s", e, exc_info=True)
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
