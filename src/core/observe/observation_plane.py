import asyncio
import logging
from typing import Any, Dict

from src.core.observe.event_bus import EventBus
from src.core.observe.package_inventory import PackageInventory
from src.core.observe.snapshot_api import SnapshotAPI
from src.core.observe.system_collector import SystemCollector
from src.core.observe.window_tracker import WindowTracker

logger = logging.getLogger(__name__)


class ObservationPlane:
    """Unified Orchestrator for all system observation, telemetry, and event stream pipelines."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        logger.info("Initializing Observation Plane...")
        self.events = EventBus()
        self.system = SystemCollector()
        self.snapshot = SnapshotAPI()
        self.packages = PackageInventory()
        self.windows = WindowTracker()
        self._initialized = True
        logger.info("✅ Observation Plane successfully initialized")

    async def get_integrated_state(self) -> Dict[str, Any]:
        """Aggregate data from all observers to generate a comprehensive system state report."""
        try:
            metrics_task = self.system.get_all_metrics()
            snapshot_task = self.snapshot.get_system_snapshot()
            packages_task = self.packages.get_unified_inventory()
            window_task = self.windows.get_active_window()

            metrics, snapshot, packages, active_window = await asyncio.gather(
                metrics_task, snapshot_task, packages_task, window_task
            )

            return {
                "metrics": metrics,
                "snapshot": snapshot,
                "packages_summary": {
                    mgr: len(pkgs) for mgr, pkgs in packages.items()
                },
                "active_window": active_window,
            }
        except Exception as e:
            logger.error(f"ObservationPlane: Failed to compile integrated system state: {e}")
            return {"error": str(e)}

    async def trigger_event_capture(self, event_type: str, custom_data: Any = None):
        """Manually trigger a metrics snapshot and publish it to the event bus."""
        state = await self.get_integrated_state()
        if custom_data:
            state["custom_data"] = custom_data
        await self.events.publish(event_type, state)
