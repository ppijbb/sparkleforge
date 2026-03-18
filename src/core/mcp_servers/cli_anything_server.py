"""CLI-Anything MCP Server - Run and discover CLI-Anything harnesses.

Exposes installed cli-anything-* commands as MCP tools so agents can
invoke them with structured JSON output and list available harnesses.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

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

mcp = FastMCP("cli-anything")

def _is_autonomy_mode() -> bool:
    return (
        os.getenv("SPARKLEFORGE_AUTONOMY_MODE", "false")
        .strip()
        .lower()
        in {"1", "true", "yes", "y", "on"}
    )


def _env_csv(name: str) -> List[str]:
    value = os.getenv(name, "") or ""
    items = [x.strip() for x in value.split(",")]
    return [x for x in items if x]


def _validate_cli_anything_autonomy(software: str, command: str) -> str | None:
    """Autonomy mode에서 cli-anything 실행을 allowlist로 제한.

    정책:
    - `CLI_ANYTHING_ALLOWED_SOFTWARES`가 비어 있으면(=사용자 미설정) cli-anything 자체를 deny
    - `CLI_ANYTHING_COMMAND_ALLOW_REGEXES`가 비어 있으면(=사용자 미설정) command deny
    """
    if not _is_autonomy_mode():
        return None

    allowed_softwares = _env_csv("CLI_ANYTHING_ALLOWED_SOFTWARES")
    if not allowed_softwares:
        return "Autonomy mode: CLI_ANYTHING_ALLOWED_SOFTWARES is not set (deny by default)"
    if "*" not in allowed_softwares and software not in allowed_softwares:
        return f"Autonomy mode: software '{software}' is not in CLI_ANYTHING_ALLOWED_SOFTWARES"

    allowed_regexes = _env_csv("CLI_ANYTHING_COMMAND_ALLOW_REGEXES")
    if not allowed_regexes:
        return "Autonomy mode: CLI_ANYTHING_COMMAND_ALLOW_REGEXES is not set (deny by default)"

    cmd = (command or "").strip()
    if not cmd:
        return "Autonomy mode: empty command is not allowed"

    for rx in allowed_regexes:
        try:
            if re.search(rx, cmd, flags=re.IGNORECASE):
                return None
        except re.error:
            # 잘못된 정규식은 운영자가 수정해야 하므로 deny에 포함
            continue

    return "Autonomy mode: command does not match any CLI_ANYTHING_COMMAND_ALLOW_REGEXES"


class RunCliAnythingInput(BaseModel):
    """Input for running a CLI-Anything harness command."""

    software: str = Field(
        ...,
        description="Software name (e.g. gimp, blender, libreoffice). Must match installed cli-anything-<software>.",
        min_length=1,
        max_length=64,
    )
    command: str = Field(
        ...,
        description="Subcommand and arguments as a single string, e.g. 'project new -o out.json' or 'document info --project doc.json'.",
        min_length=0,
        max_length=2048,
    )
    use_json: bool = Field(
        default=True,
        description="Prepend --json for machine-readable output. Set false for human-readable output.",
    )
    timeout_seconds: int = Field(
        default=120,
        description="Maximum execution time in seconds.",
        ge=1,
        le=600,
    )


def _cli_binary_name(software: str) -> str:
    """Return the CLI binary name for a software (e.g. cli-anything-gimp)."""
    base = software.strip().lower().replace("_", "-")
    return f"cli-anything-{base}"


def _find_cli_anything_commands() -> List[Dict[str, str]]:
    """Discover cli-anything-* commands available in PATH."""
    path_env = os.environ.get("PATH", "")
    path_dirs = [p.strip() for p in path_env.split(os.pathsep) if p.strip()]
    seen: set = set()
    result: List[Dict[str, str]] = []

    for directory in path_dirs:
        try:
            entries = os.listdir(directory)
        except (OSError, PermissionError):
            continue
        for name in entries:
            if name.startswith("cli-anything-") and name not in seen:
                full = os.path.join(directory, name)
                if os.path.isfile(full) and os.access(full, os.X_OK):
                    seen.add(name)
                    software = name.replace("cli-anything-", "", 1)
                    result.append({"name": name, "software": software, "path": full})

    result.sort(key=lambda x: x["name"])
    return result


def _run_cli_anything_sync(
    software: str,
    command: str,
    use_json: bool = True,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Synchronously run cli-anything-<software> and return result dict."""
    cli_name = _cli_binary_name(software)
    path = shutil.which(cli_name)
    if not path:
        return {
            "success": False,
            "error": f"CLI not found: {cli_name}. Install with: pip install -e . in the software's agent-harness directory.",
            "software": software,
            "command": command,
        }

    parts = [path]
    if use_json:
        parts.append("--json")
    if command.strip():
        parts.extend(command.strip().split())

    try:
        proc = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout_seconds}s",
            "software": software,
            "command": command,
        }
    except Exception as e:
        logger.exception("run_cli_anything subprocess error")
        return {
            "success": False,
            "error": str(e),
            "software": software,
            "command": command,
        }

    out = proc.stdout or ""
    err = proc.stderr or ""

    if use_json and out.strip():
        try:
            parsed = json.loads(out)
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "software": software,
                "command": command,
                "data": parsed,
                "stderr": err.strip() or None,
            }
        except json.JSONDecodeError:
            pass

    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "software": software,
        "command": command,
        "stdout": out,
        "stderr": err.strip() or None,
    }


@mcp.tool()
async def run_cli_anything(input: RunCliAnythingInput) -> str:
    """Run a CLI-Anything harness command and return structured result.

    Use this after a cli-anything harness is installed (e.g. cli-anything-gimp).
    Pass the software name and the subcommand + args as a single string.
    By default --json is prepended for machine-readable output.

    Returns JSON with: success, returncode, software, command, and either
    data (parsed JSON from CLI) or stdout/stderr.
    """
    denial = _validate_cli_anything_autonomy(input.software, input.command)
    if denial is not None:
        return json.dumps(
            {
                "success": False,
                "error": denial,
                "software": input.software,
                "command": input.command,
            },
            ensure_ascii=False,
            indent=2,
        )

    result = await asyncio.to_thread(
        _run_cli_anything_sync,
        input.software,
        input.command,
        input.use_json,
        input.timeout_seconds,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_cli_anything_tools() -> str:
    """List all installed CLI-Anything harnesses available in PATH.

    Returns JSON array of objects with: name (e.g. cli-anything-gimp),
    software (e.g. gimp), path (full path to executable).
    Use these software names with run_cli_anything.
    """
    tools = await asyncio.to_thread(_find_cli_anything_commands)
    return json.dumps(
        {"tools": tools, "count": len(tools)},
        ensure_ascii=False,
        indent=2,
    )


def run() -> None:
    """Run the CLI-Anything MCP server."""
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run()
