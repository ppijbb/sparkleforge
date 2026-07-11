import abc
import asyncio
import json
import logging
import shlex
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional
import websockets

from src.core.trust_gate import TrustContext

logger = logging.getLogger(__name__)


class RemoteSession(abc.ABC):
    """Abstract base class representing a remote node session."""

    def __init__(self, trust_context: Optional[TrustContext] = None):
        self.trust_context = trust_context or TrustContext.default()

    @abc.abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the remote node."""
        pass

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the remote node."""
        pass

    @abc.abstractmethod
    async def execute(self, command: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Execute a shell command on the remote node."""
        pass

    async def send_trust_context(self, trust: TrustContext) -> bool:
        """Update and propagate trust context to the remote node."""
        self.trust_context = trust
        return True

    @abc.abstractmethod
    async def send_payload(self, action: str, payload: Dict[str, Any]) -> bool:
        """Send a structured control payload (memory sync, credential handoff, etc.) to the remote node."""
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Return connection status."""
        pass


class WebSocketRemoteSession(RemoteSession):
    """WebSocket-based remote session with automatic reconnection and backoff."""

    def __init__(
        self,
        url: str,
        trust_context: Optional[TrustContext] = None,
        reconnect_interval_base: float = 2.0,
        reconnect_max_interval: float = 30.0,
        reconnect_backoff_factor: float = 2.0,
    ):
        super().__init__(trust_context)
        self.url = url
        self.reconnect_interval_base = reconnect_interval_base
        self.reconnect_max_interval = reconnect_max_interval
        self.reconnect_backoff_factor = reconnect_backoff_factor

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._should_reconnect = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        self._should_reconnect = True
        success = await self._connect_internal()
        if not success:
            async with self._lock:
                if self._reconnect_task is None or self._reconnect_task.done():
                    self._reconnect_task = asyncio.create_task(self._monitor_connection())
        return success

    async def _connect_internal(self) -> bool:
        async with self._lock:
            if self._connected:
                return True
            try:
                self._ws = await websockets.connect(self.url)
                self._connected = True
                logger.info(f"WebSocket remote session connected to {self.url}")
                # Start background reconnection monitor if not active
                if self._reconnect_task is None or self._reconnect_task.done():
                    self._reconnect_task = asyncio.create_task(self._monitor_connection())
                return True
            except Exception as e:
                logger.error(f"Failed to connect to WebSocket {self.url}: {e}")
                self._connected = False
                return False

    async def disconnect(self) -> None:
        self._should_reconnect = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        async with self._lock:
            if self._ws:
                await self._ws.close()
                self._ws = None
            self._connected = False
            logger.info("WebSocket remote session disconnected.")

    async def _monitor_connection(self) -> None:
        """Background loop to detect disconnection and trigger reconnection with backoff."""
        attempt = 0
        while self._should_reconnect:
            if not self._connected or (self._ws and self._ws.closed):
                self._connected = False
                wait_time = min(
                    self.reconnect_max_interval,
                    self.reconnect_interval_base * (self.reconnect_backoff_factor ** attempt),
                )
                logger.warning(
                    f"WebSocket disconnected. Reconnecting in {wait_time:.1f}s (attempt {attempt + 1})..."
                )
                await asyncio.sleep(wait_time)
                success = await self._connect_internal()
                if success:
                    attempt = 0  # reset backoff on successful connection
                else:
                    attempt += 1
            else:
                await asyncio.sleep(1.0)

    async def execute(self, command: str, timeout: float = 30.0) -> Dict[str, Any]:
        if not self._connected or not self._ws:
            return {
                "stdout": "",
                "stderr": "Remote session not connected.",
                "returncode": -1,
                "status": "failed",
            }

        payload = {
            "action": "execute",
            "command": command,
            "trust_context": self.trust_context.to_dict(),
        }

        try:
            async with asyncio.timeout(timeout):
                await self._ws.send(json.dumps(payload))
                response_raw = await self._ws.recv()
                response = json.loads(response_raw)
                return response
        except asyncio.TimeoutError:
            logger.error(f"WebSocket execution timeout ({timeout}s) for command: {command}")
            return {
                "stdout": "",
                "stderr": f"WebSocket execution timed out after {timeout} seconds.",
                "returncode": -1,
                "status": "timeout",
            }
        except Exception as e:
            logger.error(f"WebSocket execution error: {e}")
            self._connected = False  # trigger reconnection
            return {
                "stdout": "",
                "stderr": f"WebSocket connection error: {e}",
                "returncode": -1,
                "status": "failed",
            }

    async def send_trust_context(self, trust: TrustContext) -> bool:
        await super().send_trust_context(trust)
        return await self.send_payload("update_trust", {"trust_context": trust.to_dict()})

    async def send_payload(self, action: str, payload: Dict[str, Any]) -> bool:
        if not self._connected or not self._ws:
            return False

        message = {"action": action, **payload}
        try:
            await self._ws.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send '{action}' payload over WebSocket: {e}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None and not self._ws.closed


class SSHRemoteSession(RemoteSession):
    """SSH and Teleport-based remote session (stateless executions via subprocess)."""

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: Optional[str] = None,
        key_path: Optional[str] = None,
        use_teleport: bool = False,
        teleport_node: Optional[str] = None,
        trust_context: Optional[TrustContext] = None,
    ):
        super().__init__(trust_context)
        self.host = host
        self.port = port
        self.username = username
        self.key_path = key_path
        self.use_teleport = use_teleport
        self.teleport_node = teleport_node
        self._connected = False

    async def connect(self) -> bool:
        # For SSH, we just verify that we can execute a simple check (e.g. echo)
        logger.info(f"Connecting to SSH remote session (Teleport={self.use_teleport})...")
        res = await self.execute("echo -n 'connected'", timeout=10.0)
        self._connected = (res.get("stdout") == "connected")
        return self._connected

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("SSH remote session disconnected.")

    async def send_payload(self, action: str, payload: Dict[str, Any]) -> bool:
        """Deliver a control payload by writing JSON into the remote node's sync inbox."""
        # Action becomes part of the remote file path — restrict to safe characters
        if not action or not all(c.isalnum() or c in "_-" for c in action):
            logger.error(f"Rejected send_payload with unsafe action name: {action!r}")
            return False
        message = json.dumps({"action": action, **payload})
        inbox_dir = "~/.sparkleforge/sync"
        command = (
            f"mkdir -p {inbox_dir} && "
            f"printf '%s' {shlex.quote(message)} > {inbox_dir}/{action}.json"
        )
        res = await self.execute(command, timeout=15.0)
        return res.get("status") == "success"

    async def execute(self, command: str, timeout: float = 30.0) -> Dict[str, Any]:
        # Propagate trust context by injecting it into environment before execution
        trust_json = json.dumps(self.trust_context.to_dict())
        # Use shlex.quote for safe shell escaping of the JSON trust context payload.
        escaped_trust = shlex.quote(trust_json)

        # Build command that exports the trust context environment variable
        wrapped_command = f"export SPARKLEFORGE_TRUST_CONTEXT={escaped_trust}; {command}"

        # Construct execution argument list
        if self.use_teleport:
            node = self.teleport_node or self.host
            args = ["tsh", "ssh", node, wrapped_command]
        else:
            args = ["ssh", "-o", "ConnectTimeout=10"]
            if self.port != 22:
                args.extend(["-p", str(self.port)])
            if self.key_path:
                args.extend(["-i", self.key_path])
            
            target = f"{self.username}@{self.host}" if self.username else self.host
            args.extend([target, wrapped_command])

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                stdout = stdout_bytes.decode(errors="replace")
                stderr = stderr_bytes.decode(errors="replace")
                returncode = proc.returncode
                status = "success" if returncode == 0 else "failed"
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode,
                    "status": status,
                }
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                logger.error(f"SSH execution timeout ({timeout}s) for command: {command}")
                return {
                    "stdout": "",
                    "stderr": f"SSH command execution timed out after {timeout} seconds.",
                    "returncode": -1,
                    "status": "timeout",
                }
        except Exception as e:
            logger.error(f"SSH execution process execution failed: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "status": "failed",
            }

    @property
    def is_connected(self) -> bool:
        return self._connected


class SessionManager:
    """Manages persistent remote session states in SQLite.

    Supports serialization, recovery, pause/resume status updates, and
    webhook notifications for HITL checkpoints.
    """

    def __init__(self, db_path: str = "sessions.db", webhook_url: Optional[str] = None):
        self.db_path = db_path
        self.webhook_url = webhook_url
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def save_session(
        self, session_id: str, state: Dict[str, Any], status: str = "active"
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)",
                (session_id, json.dumps(state), status, time.time()),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def get_session_status(self, session_id: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row[0] if row else None

    def list_sessions(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, status, updated_at FROM sessions"
            ).fetchall()
            return [
                {"session_id": r[0], "status": r[1], "updated_at": r[2]}
                for r in rows
            ]

    def update_status(self, session_id: str, status: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (status, time.time(), session_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    async def notify_webhook(
        self, checkpoint_id: str, stage: str, payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        if not self.webhook_url:
            return False
        import aiohttp

        body = {
            "checkpoint_id": checkpoint_id,
            "stage": stage,
            "event": "hitl_checkpoint",
            "timestamp": time.time(),
        }
        if payload:
            body.update(payload)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=body) as resp:
                    return resp.status < 400
        except Exception as e:
            logger.error("Failed to send webhook notification: %s", e)
            return False
