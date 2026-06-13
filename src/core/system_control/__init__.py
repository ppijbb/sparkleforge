"""Read-only system inventory and guarded control foundations."""

from .inventory_store import InventoryStore, collect_inventory

__all__ = ["InventoryStore", "collect_inventory"]
