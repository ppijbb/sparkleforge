import logging
from typing import Any, Dict, List

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_RESOURCE_THRESHOLDS = {
    "cpu_percent": 90.0,
    "memory_percent": 90.0,
    "disk_percent": 90.0,
}


class SystemCollector:
    """Collects hardware and OS resource metrics using psutil."""

    def __init__(self, thresholds: Dict[str, float] | None = None):
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil is not installed. SystemCollector will return mock metrics.")
        self.thresholds = {**DEFAULT_RESOURCE_THRESHOLDS, **(thresholds or {})}

    async def get_cpu_info(self) -> Dict[str, Any]:
        """Fetch CPU usage and frequency metrics."""
        if not PSUTIL_AVAILABLE:
            return {"percent": 0.0, "count": 1, "frequency_mhz": 0.0}

        try:
            return {
                "percent": psutil.cpu_percent(interval=None),
                "count": psutil.cpu_count(logical=True),
                "physical_count": psutil.cpu_count(logical=False),
                "load_avg": getattr(psutil, "getloadavg", lambda: (0.0, 0.0, 0.0))(),
            }
        except Exception as e:
            logger.error(f"SystemCollector: Failed to fetch CPU info: {e}")
            return {"error": str(e)}

    async def get_memory_info(self) -> Dict[str, Any]:
        """Fetch RAM and swap memory metrics."""
        if not PSUTIL_AVAILABLE:
            return {"total_bytes": 0, "available_bytes": 0, "percent": 0.0}

        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "total_bytes": mem.total,
                "available_bytes": mem.available,
                "used_bytes": mem.used,
                "percent": mem.percent,
                "swap_total_bytes": swap.total,
                "swap_percent": swap.percent,
            }
        except Exception as e:
            logger.error(f"SystemCollector: Failed to fetch memory info: {e}")
            return {"error": str(e)}

    async def get_disk_info(self) -> Dict[str, Any]:
        """Fetch disk space usage for the root partition."""
        if not PSUTIL_AVAILABLE:
            return {"total_bytes": 0, "free_bytes": 0, "percent": 0.0}

        try:
            usage = psutil.disk_usage("/")
            return {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "percent": usage.percent,
            }
        except Exception as e:
            logger.error(f"SystemCollector: Failed to fetch disk info: {e}")
            return {"error": str(e)}

    async def get_network_info(self) -> Dict[str, Any]:
        """Fetch cumulative network IO statistics."""
        if not PSUTIL_AVAILABLE:
            return {"bytes_sent": 0, "bytes_recv": 0}

        try:
            io = psutil.net_io_counters()
            return {
                "bytes_sent": io.bytes_sent,
                "bytes_recv": io.bytes_recv,
                "packets_sent": io.packets_sent,
                "packets_recv": io.packets_recv,
                "errin": io.errin,
                "errout": io.errout,
            }
        except Exception as e:
            logger.error(f"SystemCollector: Failed to fetch network info: {e}")
            return {"error": str(e)}

    async def get_battery_info(self) -> Dict[str, Any]:
        """Fetch battery charge and power plug status."""
        if not PSUTIL_AVAILABLE:
            return {"percent": 100.0, "power_plugged": True}

        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return {"status": "no battery detected"}
            return {
                "percent": battery.percent,
                "secsleft": battery.secsleft,
                "power_plugged": battery.power_plugged,
            }
        except Exception as e:
            logger.debug(f"SystemCollector: Failed to fetch battery info: {e}")
            return {"error": str(e)}

    async def get_temperature_info(self) -> Dict[str, Any]:
        """Fetch sensor temperatures (if supported)."""
        if not PSUTIL_AVAILABLE or not hasattr(psutil, "sensors_temperatures"):
            return {"status": "temperature sensors unsupported"}

        try:
            temps = psutil.sensors_temperatures()
            result = {}
            for name, entries in temps.items():
                result[name] = [
                    {"label": entry.label, "current": entry.current, "high": entry.high, "critical": entry.critical}
                    for entry in entries
                ]
            return result if result else {"status": "no temperature metrics recorded"}
        except Exception as e:
            logger.debug(f"SystemCollector: Failed to fetch temperature info: {e}")
            return {"error": str(e)}

    async def get_all_metrics(self) -> Dict[str, Any]:
        """Fetch all hardware resource metrics in a unified dictionary."""
        return {
            "cpu": await self.get_cpu_info(),
            "memory": await self.get_memory_info(),
            "disk": await self.get_disk_info(),
            "network": await self.get_network_info(),
            "battery": await self.get_battery_info(),
            "temperature": await self.get_temperature_info(),
        }

    def check_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """Compare a get_all_metrics() snapshot against configured thresholds.

        Returns a list of human-readable warnings for any metric that
        exceeds its threshold (empty list when everything is within bounds).
        Errors in an individual metric are ignored here — get_*_info already
        logs those; this only judges values that were collected successfully.
        """
        warnings: List[str] = []

        cpu_percent = metrics.get("cpu", {}).get("percent")
        if isinstance(cpu_percent, (int, float)) and cpu_percent > self.thresholds["cpu_percent"]:
            warnings.append(
                f"CPU usage at {cpu_percent:.1f}% (threshold {self.thresholds['cpu_percent']:.1f}%)"
            )

        memory_percent = metrics.get("memory", {}).get("percent")
        if (
            isinstance(memory_percent, (int, float))
            and memory_percent > self.thresholds["memory_percent"]
        ):
            warnings.append(
                f"Memory usage at {memory_percent:.1f}% "
                f"(threshold {self.thresholds['memory_percent']:.1f}%)"
            )

        disk_percent = metrics.get("disk", {}).get("percent")
        if isinstance(disk_percent, (int, float)) and disk_percent > self.thresholds["disk_percent"]:
            warnings.append(
                f"Disk usage at {disk_percent:.1f}% (threshold {self.thresholds['disk_percent']:.1f}%)"
            )

        return warnings


def check_disk_space_safety(min_free_mb: float = 500.0, path: str = "/") -> tuple[bool, str]:
    """Pre-flight check: is there enough free disk space to safely start a run.

    Low disk space can cause SQLite writes to fail mid-transaction (and
    leave a locked/corrupt database), so callers should reject the run
    rather than let it fail unpredictably partway through.

    Returns (is_safe, message). is_safe is True (with an empty message)
    when psutil is unavailable, since we'd rather run unchecked than
    block on a check we can't actually perform.
    """
    if not PSUTIL_AVAILABLE:
        return True, ""

    try:
        free_mb = psutil.disk_usage(path).free / (1024 * 1024)
    except Exception as e:
        logger.warning(f"check_disk_space_safety: failed to read disk usage: {e}")
        return True, ""

    if free_mb < min_free_mb:
        return False, (
            f"Only {free_mb:.0f}MB free on '{path}' (safety threshold {min_free_mb:.0f}MB). "
            "Free up disk space before starting a run — low disk space can cause "
            "SQLite writes to fail mid-transaction."
        )
    return True, ""


def check_network_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> tuple[bool, str]:
    """Pre-flight check: is there basic network connectivity before scheduling LLM calls.

    A single fast TCP probe, not a guarantee any specific provider endpoint
    is reachable — just enough to warn upfront on a fully offline host
    instead of discovering it only after several provider socket timeouts
    have each run their full course.

    Returns (is_connected, message). Warn-only by design: some setups run
    entirely against local models and don't need internet access, and a
    restrictive network that blocks this specific probe host doesn't mean
    every provider is actually unreachable.
    """
    import socket

    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True, ""
    except OSError as e:
        return False, (
            f"No network connectivity detected ({e}). Remote LLM providers will "
            "fail/time out without a network connection; local-model-only setups "
            "can ignore this."
        )
