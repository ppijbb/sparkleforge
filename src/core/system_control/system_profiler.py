"""Read-only hardware and OS resource profiling."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any

import psutil


def _run(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _disk_partitions() -> list[dict[str, Any]]:
    partitions: list[dict[str, Any]] = []
    for part in psutil.disk_partitions(all=False):
        entry: dict[str, Any] = {
            "device": part.device,
            "mountpoint": part.mountpoint,
            "fstype": part.fstype,
        }
        try:
            usage = psutil.disk_usage(part.mountpoint)
            entry["total_bytes"] = usage.total
            entry["used_bytes"] = usage.used
            entry["free_bytes"] = usage.free
            entry["percent"] = usage.percent
        except (OSError, PermissionError):
            entry["usage_error"] = "unavailable"
        partitions.append(entry)
    return partitions


def _network_interfaces() -> list[dict[str, Any]]:
    stats = psutil.net_if_stats()
    interfaces: list[dict[str, Any]] = []
    for name, addrs in psutil.net_if_addrs().items():
        interfaces.append(
            {
                "name": name,
                "is_up": stats.get(name).isup if name in stats else None,
                "speed_mbps": stats.get(name).speed if name in stats else None,
                "addresses": [
                    {
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": addr.netmask,
                    }
                    for addr in addrs
                ],
            }
        )
    return interfaces


def _gpu_info() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi"):
        output = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        if output:
            gpus = []
            for line in output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                gpus.append(
                    {
                        "vendor": "nvidia",
                        "name": parts[0] if parts else "",
                        "memory_total": parts[1] if len(parts) > 1 else None,
                        "driver_version": parts[2] if len(parts) > 2 else None,
                    }
                )
            return gpus
    if shutil.which("rocm-smi"):
        output = _run(["rocm-smi", "--showproductname"])
        if output:
            return [{"vendor": "amd", "raw": output}]
    return []


def _usb_devices() -> list[str]:
    if not shutil.which("lsusb"):
        return []
    output = _run(["lsusb"])
    return output.splitlines() if output else []


def _sensors() -> dict[str, Any]:
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except (AttributeError, OSError):
        temps = {}
    try:
        fans = psutil.sensors_fans()
    except (AttributeError, OSError):
        fans = {}
    return {
        "temperatures": {
            key: [entry._asdict() for entry in value] for key, value in temps.items()
        },
        "fans": {key: [entry._asdict() for entry in value] for key, value in fans.items()},
    }


def collect_hardware_profile() -> dict[str, Any]:
    """Collect a read-only hardware profile for the current machine."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "cpu": {
            "model": platform.processor() or platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "architecture": platform.machine(),
        },
        "memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "percent": memory.percent,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
        },
        "disks": _disk_partitions(),
        "gpus": _gpu_info(),
        "network_interfaces": _network_interfaces(),
        "usb_devices": _usb_devices(),
        "sensors": _sensors(),
    }
