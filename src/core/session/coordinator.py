import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.session.remote_session import RemoteSession
from src.core.session.secure_envelope import (
    decrypt_credential_envelope,
    encrypt_credential_envelope,
)
from src.core.trust_gate import TrustContext
from src.core.guard.credential_vault import CredentialVault
from src.core.guard.guard_plane import GuardPlane
from src.core.guard.credential_vault import CredentialVault

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


class CoordinatorNode:
    """Coordinator Node managing multiple Worker Nodes, pairing, heartbeat, and failover."""

    def __init__(
        self,
        guard_plane: Optional[GuardPlane] = None,
        vault: Optional[CredentialVault] = None,
    ):
        self.guard_plane = guard_plane or GuardPlane()
        # Shared, caller-configured vault. Intentionally no default instance:
        # an implicitly created vault would be empty and turn a configuration
        # error into an opaque retrieval failure at delegation time.
        self._vault = vault
        self.active_workers: Dict[str, RemoteSession] = {}
        self.worker_statuses: Dict[str, NodeStatus] = {}
        self.worker_loads: Dict[str, int] = {}
        self.heartbeat_failures: Dict[str, int] = {}
        # Per-worker pairing secrets used to seal credential envelopes
        self.worker_secrets: Dict[str, str] = {}
        
        # Track tasks: task_id -> worker_id
        self.task_assignments: Dict[str, str] = {}
        # Store delegated task payloads for failover recovery: task_id -> task_payload
        self.task_payloads: Dict[str, Dict[str, Any]] = {}
        
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._round_robin_index = 0

    def register_worker(
        self,
        worker_id: str,
        session: RemoteSession,
        shared_secret: Optional[str] = None,
    ) -> None:
        """Register (pair) a new worker node.

        ``shared_secret`` is the pairing secret used to encrypt credential
        envelopes for this worker; without it, credential delegation to the
        worker is refused.
        """
        self.active_workers[worker_id] = session
        self.worker_statuses[worker_id] = NodeStatus.ONLINE
        self.worker_loads[worker_id] = 0
        self.heartbeat_failures[worker_id] = 0
        if shared_secret:
            self.worker_secrets[worker_id] = shared_secret
        logger.info(f"Worker node '{worker_id}' paired and registered successfully.")

    async def discover_workers(
        self,
        candidates: Dict[str, RemoteSession],
        probe_timeout: float = 5.0,
    ) -> List[str]:
        """Probe candidate node sessions and auto-pair those that respond.

        Candidates already registered are skipped; unresponsive candidates are
        left unregistered so discovery can be retried later.
        """
        async def probe(worker_id: str, session: RemoteSession) -> Optional[str]:
            try:
                connected = await asyncio.wait_for(session.connect(), timeout=probe_timeout)
            except Exception as e:
                logger.warning(f"Discovery probe failed for candidate '{worker_id}': {e}")
                return None
            if connected:
                self.register_worker(worker_id, session)
                return worker_id
            logger.info(f"Candidate node '{worker_id}' did not respond to discovery probe.")
            return None

        results = await asyncio.gather(*(
            probe(worker_id, session)
            for worker_id, session in candidates.items()
            if worker_id not in self.active_workers
        ))
        return [worker_id for worker_id in results if worker_id is not None]

    def deregister_worker(self, worker_id: str) -> None:
        """Deregister a worker node."""
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
        if worker_id in self.worker_statuses:
            del self.worker_statuses[worker_id]
        if worker_id in self.worker_loads:
            del self.worker_loads[worker_id]
        if worker_id in self.heartbeat_failures:
            del self.heartbeat_failures[worker_id]
        if worker_id in self.worker_secrets:
            del self.worker_secrets[worker_id]
        logger.info(f"Worker node '{worker_id}' deregistered.")

    def start_heartbeat_loop(self, interval: float = 2.0, max_failures: int = 3) -> None:
        """Start background heartbeat checks."""
        self._is_running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval, max_failures))
        logger.info("Coordinator heartbeat monitoring started.")

    async def stop_heartbeat_loop(self) -> None:
        """Stop background heartbeat checks."""
        self._is_running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        logger.info("Coordinator heartbeat monitoring stopped.")

    async def _heartbeat_loop(self, interval: float, max_failures: int) -> None:
        while self._is_running:
            await asyncio.sleep(interval)
            await self._check_heartbeats(max_failures)

    async def _check_heartbeats(self, max_failures: int) -> None:
        """Send heartbeats (ping) to all registered worker nodes."""
        for worker_id, session in list(self.active_workers.items()):
            if self.worker_statuses.get(worker_id) == NodeStatus.OFFLINE:
                continue

            try:
                # Use a short timeout for heartbeats
                res = await asyncio.wait_for(session.execute("ping"), timeout=1.0)
                if res.get("status") == "success" or res.get("stdout") == "pong":
                    self.heartbeat_failures[worker_id] = 0
                else:
                    self._handle_heartbeat_failure(worker_id, max_failures)
            except Exception as e:
                logger.warning(f"Heartbeat failed for worker '{worker_id}': {e}")
                self._handle_heartbeat_failure(worker_id, max_failures)

    def _handle_heartbeat_failure(self, worker_id: str, max_failures: int) -> None:
        self.heartbeat_failures[worker_id] = self.heartbeat_failures.get(worker_id, 0) + 1
        failures = self.heartbeat_failures[worker_id]
        logger.warning(f"Worker '{worker_id}' heartbeat failure count: {failures}/{max_failures}")
        
        if failures >= max_failures:
            logger.error(f"Worker '{worker_id}' reached max heartbeat failures. Marking OFFLINE.")
            self.worker_statuses[worker_id] = NodeStatus.OFFLINE
            asyncio.create_task(self._handle_failover(worker_id))

    async def _handle_failover(self, offline_worker_id: str) -> None:
        """Reschedule all active tasks of the offline worker node to other online workers."""
        affected_tasks = [
            task_id for task_id, worker_id in self.task_assignments.items()
            if worker_id == offline_worker_id
        ]
        
        logger.info(f"Triggering failover for worker '{offline_worker_id}'. Affected tasks: {affected_tasks}")
        
        for task_id in affected_tasks:
            # Remove assignment
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            
            # Retrieve original task payload
            payload = self.task_payloads.get(task_id)
            if payload:
                logger.info(f"Rescheduling task '{task_id}' to a healthy node...")
                # Re-delegate task
                success = await self.delegate_task(task_id, payload)
                if not success:
                    logger.critical(f"Failed to failover task '{task_id}': No healthy nodes available.")
            else:
                logger.warning(f"No payload found for task '{task_id}', cannot failover.")

    async def delegate_task(self, task_id: str, task_payload: Dict[str, Any]) -> bool:
        """Route and execute a task on a selected active worker node (round-robin + load-balancing)."""
        online_workers = [
            wid for wid, status in self.worker_statuses.items()
            if status == NodeStatus.ONLINE
        ]
        
        if not online_workers:
            logger.error("No online worker nodes available for task delegation.")
            return False

        # 1. Select worker node (Least Load with Round Robin tie-breaker)
        selected_worker = min(online_workers, key=lambda wid: self.worker_loads.get(wid, 0))
        
        # Save payload for potential failover recovery
        self.task_payloads[task_id] = task_payload
        self.task_assignments[task_id] = selected_worker
        self.worker_loads[selected_worker] += 1
        
        session = self.active_workers[selected_worker]
        logger.info(f"Delegating task '{task_id}' to worker '{selected_worker}' (load: {self.worker_loads[selected_worker]}).")
        
        try:
            # Execute delegated task on worker
            res = await session.execute(task_payload.get("command", ""), timeout=task_payload.get("timeout", 30.0))
            self.worker_loads[selected_worker] = max(0, self.worker_loads[selected_worker] - 1)
            return res.get("status") == "success"
        except Exception as e:
            logger.error(f"Error executing delegated task '{task_id}' on worker '{selected_worker}': {e}")
            self.worker_loads[selected_worker] = max(0, self.worker_loads[selected_worker] - 1)
            # Instantly trigger failover
            await self._handle_failover(selected_worker)
            return False

    async def delegate_memory(self, worker_id: str, namespace: str, entries: Dict[str, Any]) -> bool:
        """Delegate (replicate) shared memory entries to a specific worker node."""
        session = self._get_online_session(worker_id)
        if session is None:
            return False
        ok = await session.send_payload(
            "sync_memory",
            {"namespace": namespace, "entries": entries},
        )
        if not ok:
            logger.warning(f"Memory delegation to worker '{worker_id}' failed.")
        return ok

    async def delegate_credential(
        self,
        worker_id: str,
        credential_key: str,
        ttl_seconds: float = 300.0,
        vault: Optional[CredentialVault] = None,
    ) -> bool:
        """Hand off a vault credential to a worker node with a bounded lifetime.

        The credential is read from the shared CredentialVault, sealed into an
        AES-256-GCM envelope with the worker's pairing secret, and sent with an
        absolute expiry timestamp; workers must discard it after expiry. The
        plaintext value never crosses the transport.
        """
        session = self._get_online_session(worker_id)
        if session is None:
            return False

        active_vault = vault or self._vault
        if active_vault is None:
            logger.error(
                "No CredentialVault configured for credential delegation; "
                "pass one to CoordinatorNode(vault=...) or delegate_credential(vault=...)."
            )
            return False

        shared_secret = self.worker_secrets.get(worker_id)
        if not shared_secret:
            logger.error(
                f"No pairing secret registered for worker '{worker_id}'; "
                "refusing to transmit credential without envelope encryption."
            )
            return False

        value = active_vault.retrieve(credential_key)
        if value is None:
            logger.error(f"Credential '{credential_key}' not found in vault; cannot delegate.")
            return False

        envelope = encrypt_credential_envelope(
            shared_secret, credential_key, value, time.time() + ttl_seconds
        )
        ok = await session.send_payload(
            "receive_credential",
            {"key": credential_key, "envelope": envelope},
        )
        if not ok:
            logger.warning(f"Credential delegation to worker '{worker_id}' failed.")
        return ok

    def _get_online_session(self, worker_id: str) -> Optional[RemoteSession]:
        if self.worker_statuses.get(worker_id) != NodeStatus.ONLINE:
            logger.error(f"Worker '{worker_id}' is not online; delegation refused.")
            return None
        return self.active_workers.get(worker_id)

    async def sync_policy(self, deny_names: List[str], deny_prefixes: List[str]) -> bool:
        """Broadcast updated GuardPlane capabilities/policies to all active workers."""
        logger.info("Broadcasting policy synchronization to workers...")
        success = True
        
        # Build local trust context payload
        trust = TrustContext(
            level=self.guard_plane.capability_manager.get_default_trust_level() if hasattr(self.guard_plane.capability_manager, "get_default_trust_level") else TrustContext.default().level,
            deny_names=frozenset(deny_names),
            deny_prefixes=tuple(deny_prefixes),
        )
        
        for worker_id, session in self.active_workers.items():
            if self.worker_statuses.get(worker_id) == NodeStatus.ONLINE:
                ok = await session.send_trust_context(trust)
                if not ok:
                    logger.warning(f"Failed to sync policy to worker '{worker_id}'.")
                    success = False
        return success

class WorkerNode:
    """Worker Node listening to Coordinator commands and returning results."""

    def __init__(
        self,
        worker_id: str,
        guard_plane: Optional[GuardPlane] = None,
        shared_secret: Optional[str] = None,
    ):
        self.worker_id = worker_id
        self.guard_plane = guard_plane or GuardPlane()
        # Pairing secret matching the coordinator's; needed to open credential envelopes
        self._shared_secret = shared_secret
        self.trust_context = TrustContext.default()
        # Delegated state from coordinator: namespace -> {key: value}
        self.shared_memory: Dict[str, Dict[str, Any]] = {}
        # Delegated credentials: key -> {"value": str, "expires_at": float}
        self._delegated_credentials: Dict[str, Dict[str, Any]] = {}

    async def handle_ping(self) -> Dict[str, Any]:
        """Respond to coordinator heartbeats."""
        return {"status": "success", "stdout": "pong"}

    async def handle_execute(self, command: str) -> Dict[str, Any]:
        """Execute action after security validation via local GuardPlane."""
        # 1. Block commands locally if they violate current trust context
        executable = command.strip().split()[0] if command.strip() else ""
        if not self.trust_context.allows_tool(executable):
            return {
                "stdout": "",
                "stderr": "Blocked: Command violates local worker trust policy.",
                "returncode": -1,
                "status": "failed"
            }

        # 2. Pass to local GuardPlane execution pipeline
        # Use capability check (default: local execution capability)
        res = self.guard_plane.check_and_execute(
            agent_id=self.worker_id,
            capability_name="execute_shell",
            command=command,
            description="Remote delegated task execution",
        )
        
        status = "success" if res.get("ok") else "failed"
        return {
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
            "returncode": res.get("returncode", 0),
            "status": status,
        }

    async def handle_sync_policy(self, trust_dict: Dict[str, Any]) -> bool:
        """Receive policy update from coordinator."""
        self.trust_context = TrustContext.from_dict(trust_dict)
        logger.info(f"Worker '{self.worker_id}' synchronized trust policy: {self.trust_context}")
        return True

    async def handle_sync_memory(self, namespace: str, entries: Dict[str, Any]) -> bool:
        """Receive delegated memory entries from coordinator."""
        self.shared_memory.setdefault(namespace, {}).update(entries)
        logger.info(
            f"Worker '{self.worker_id}' synchronized {len(entries)} memory entries "
            f"into namespace '{namespace}'."
        )
        return True

    async def handle_receive_credential(self, key: str, envelope: str) -> bool:
        """Receive an encrypted, time-bounded delegated credential from coordinator."""
        if not self._shared_secret:
            logger.warning(
                f"Worker '{self.worker_id}' has no pairing secret; rejecting credential '{key}'."
            )
            return False

        opened = decrypt_credential_envelope(self._shared_secret, key, envelope)
        if opened is None:
            logger.warning(f"Worker '{self.worker_id}' could not open credential envelope '{key}'.")
            return False
        value, expires_at = opened["value"], opened["expires_at"]

        now = time.time()
        # Evict expired entries so the store cannot grow unbounded
        for stale_key in [k for k, entry in self._delegated_credentials.items() if entry["expires_at"] <= now]:
            del self._delegated_credentials[stale_key]

        if expires_at <= now:
            logger.warning(f"Worker '{self.worker_id}' rejected already-expired credential '{key}'.")
            return False
        self._delegated_credentials[key] = {"value": value, "expires_at": expires_at}
        logger.info(f"Worker '{self.worker_id}' received delegated credential '{key}'.")
        return True

    def get_delegated_credential(self, key: str) -> Optional[str]:
        """Return a delegated credential value, discarding it if expired."""
        entry = self._delegated_credentials.get(key)
        if entry is None:
            return None
        if entry["expires_at"] <= time.time():
            del self._delegated_credentials[key]
            logger.info(f"Worker '{self.worker_id}' discarded expired credential '{key}'.")
            return None
        return entry["value"]
