"""Cached read-only system inventory."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .resource_locator import index_path_executables, standard_directories
from .software_inventory import collect_software_inventory
from .system_profiler import collect_hardware_profile


def collect_inventory() -> dict[str, Any]:
    """Collect the current read-only system inventory."""
    return {
        "schema_version": 1,
        "collected_at": time.time(),
        "hardware": collect_hardware_profile(),
        "software": collect_software_inventory(),
        "resources": {
            "standard_directories": standard_directories(),
            "path_executables": index_path_executables(limit=500),
        },
    }


class InventoryStore:
    """JSON cache for system inventory with TTL refresh."""

    def __init__(self, path: str | Path | None = None, ttl_seconds: int = 3600):
        self.path = Path(path or ".sparkleforge/inventory/system_inventory.json")
        self.ttl_seconds = ttl_seconds

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def is_fresh(self, data: dict[str, Any] | None) -> bool:
        if not data:
            return False
        collected_at = data.get("collected_at")
        if not isinstance(collected_at, (int, float)):
            return False
        return time.time() - collected_at < self.ttl_seconds

    def get(self, *, force_refresh: bool = False) -> dict[str, Any]:
        cached = self.load()
        if not force_refresh and self.is_fresh(cached):
            return cached or {}
        data = collect_inventory()
        self.save(data)
        return data
