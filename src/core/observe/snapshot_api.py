import logging
import subprocess
from typing import Any, Dict, List

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class SnapshotAPI:
    """Provides point-in-time snapshots of system processes, network ports, sessions, and services."""

    def __init__(self):
        pass

    async def get_process_snapshot(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve a list of active processes sorted by memory usage."""
        if not PSUTIL_AVAILABLE:
            return []

        processes = []
        try:
            for proc in psutil.process_iter(["pid", "name", "username", "cpu_percent", "memory_percent"]):
                try:
                    info = proc.info
                    if info["cpu_percent"] is None:
                        info["cpu_percent"] = 0.0
                    if info["memory_percent"] is None:
                        info["memory_percent"] = 0.0
                    processes.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            # Sort by memory percent decreasing
            processes.sort(key=lambda x: x.get("memory_percent", 0.0), reverse=True)
            return processes[:limit]
        except Exception as e:
            logger.error(f"SnapshotAPI: Failed to fetch process snapshot: {e}")
            return []

    async def get_port_snapshot(self) -> List[Dict[str, Any]]:
        """Retrieve active TCP/UDP listening connections."""
        if not PSUTIL_AVAILABLE:
            return []

        ports = []
        try:
            import socket
            connections = psutil.net_connections(kind="inet")
            for conn in connections:
                if conn.status == "LISTEN" or conn.type == socket.SOCK_DGRAM:
                    ports.append({
                        "fd": conn.fd,
                        "family": int(conn.family),
                        "type": int(conn.type),
                        "local_address": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                        "remote_address": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                        "status": conn.status,
                        "pid": conn.pid,
                    })
            return ports
        except Exception as e:
            logger.error(f"SnapshotAPI: Failed to fetch port snapshot: {e}")
            return []

    async def get_login_sessions(self) -> List[Dict[str, Any]]:
        """Retrieve currently logged in users."""
        if not PSUTIL_AVAILABLE:
            return []

        try:
            users = psutil.users()
            return [
                {
                    "name": u.name,
                    "terminal": u.terminal,
                    "host": u.host,
                    "started": u.started,
                    "pid": u.pid,
                }
                for u in users
            ]
        except Exception as e:
            logger.error(f"SnapshotAPI: Failed to fetch login sessions: {e}")
            return []

    async def get_service_snapshot(self) -> List[Dict[str, Any]]:
        """Fetch status of active OS services. Supported on Linux (systemd) or falls back."""
        services = []
        try:
            import shutil
            if shutil.which("systemctl"):
                proc_res = subprocess.run(
                    ["systemctl", "list-units", "--type=service", "--state=running", "--no-legend", "--all"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in proc_res.stdout.splitlines():
                    parts = line.strip().split(None, 4)
                    if len(parts) >= 4:
                        services.append({
                            "name": parts[0],
                            "load": parts[1],
                            "active": parts[2],
                            "sub": parts[3],
                            "description": parts[4] if len(parts) > 4 else "",
                        })
            else:
                services.append({"status": "systemctl unsupported on this platform"})
            return services
        except Exception as e:
            logger.debug(f"SnapshotAPI: Failed to fetch service snapshot: {e}")
            return [{"status": "failed to list services", "error": str(e)}]

    async def get_system_snapshot(self) -> Dict[str, Any]:
        """Aggregate all snapshots into a single point-in-time state telemetry dict."""
        return {
            "processes": await self.get_process_snapshot(),
            "ports": await self.get_port_snapshot(),
            "sessions": await self.get_login_sessions(),
            "services": await self.get_service_snapshot(),
        }
