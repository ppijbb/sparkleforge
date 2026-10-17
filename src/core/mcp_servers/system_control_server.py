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
