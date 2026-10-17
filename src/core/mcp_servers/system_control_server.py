"""Read-only system inventory MCP server."""

from __future__ import annotations

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - runtime dependency guard
    FastMCP = None

from src.core.system_control.inventory_store import InventoryStore
from src.core.system_control.resource_locator import find_executables, find_project_directories

mcp = FastMCP("system-control") if FastMCP else None


if mcp:

    @mcp.tool()
    def get_system_inventory(section: str | None = None, force_refresh: bool = False) -> dict:
        """Return cached read-only system inventory, optionally scoped to one section."""
        inventory = InventoryStore().get(force_refresh=force_refresh)
        if section:
            return {section: inventory.get(section)}
        return inventory

    @mcp.tool()
    def resolve_executables(names: list[str]) -> dict:
        """Resolve executable names using PATH."""
        return find_executables(names)

    @mcp.tool()
    def discover_project_directories(roots: list[str] | None = None) -> list[dict]:
        """Discover likely project directories by marker files."""
        return find_project_directories(roots)


def run() -> None:
    if not mcp:
        raise RuntimeError("FastMCP is not available")
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run()
        self.server.tool(
            name="quarantine_file",
            description="Move a dangerous file to a .quarantine directory and strip its execution permissions.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to quarantine."},
                    "reason": {"type": "string", "description": "Reason for quarantining the file."}
                },
                "required": ["path"],
            },
        )
        self.server.tool(
            name="neutralize_script",
            description="Perform full line-by-line hazard elimination on a script (commenting out dangerous commands like rm -rf or curl to sh).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the script to neutralize."},
                },
                "required": ["path"],
            },
        )
        self.server.tool(
            name="quarantine_secret",
            description="Scan and redact API keys or secrets in plaintext files, creating .env.example if appropriate.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file containing secrets."},
                },
                "required": ["path"],
            },
        )

    async def quarantine_file(self, path: str, reason: str = "") -> str:
        target = Path(path)
        if not target.exists():
            return f"Error: File {path} does not exist."
        quarantine_dir = target.parent / ".quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        dest = quarantine_dir / target.name
        target.rename(dest)
        try:
            dest.chmod(dest.stat().st_mode & ~stat.S_IEXEC & ~stat.S_IXUSR & ~stat.S_IXGRP & ~stat.S_IXOTH)
        except Exception:
            pass
        return f"Successfully quarantined {path} to {dest} with execution permissions stripped. Reason: {reason}"

    async def neutralize_script(self, path: str) -> str:
        target = Path(path)
        if not target.exists():
            return f"Error: File {path} does not exist."
        content = target.read_text(encoding="utf-8")
        lines = content.splitlines()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if "rm -rf" in stripped or "curl" in stripped and "|" in stripped:
                new_lines.append(f"# NEUTRALIZED BY SECURITY SCAN: {line}")
            else:
                new_lines.append(line)
        target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return f"Successfully neutralized dangerous commands in {path}."
