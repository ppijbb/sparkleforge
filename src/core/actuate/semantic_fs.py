import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


class PollingWatchObserver:
    """Fallback directory changes watcher using asynchronous polling."""

    def __init__(self, path: str, callback: Callable[[str, str], Any], poll_interval: float = 1.0):
        self.path = os.path.abspath(path)
        self.callback = callback
        self.poll_interval = poll_interval
        self.running = False
        self._task = None
        self._state = {}

    def _scan_directory(self) -> Dict[str, float]:
        """Scan directory and return file paths mapped to modification times."""
        state = {}
        if not os.path.exists(self.path):
            return state

        for root, _, files in os.walk(self.path):
            for file in files:
                full_path = os.path.join(root, file)
                try:
                    state[full_path] = os.path.getmtime(full_path)
                except OSError:
                    pass
        return state

    async def start(self):
        """Start the polling loop."""
        self.running = True
        self._state = self._scan_directory()
        self._task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self):
        while self.running:
            try:
                await asyncio.sleep(self.poll_interval)
                current_state = self._scan_directory()

                # Find created / modified
                for path, mtime in current_state.items():
                    if path not in self._state:
                        # Created
                        self._trigger_callback("created", path)
                    elif mtime > self._state[path]:
                        # Modified
                        self._trigger_callback("modified", path)

                # Find deleted
                for path in self._state:
                    if path not in current_state:
                        self._trigger_callback("deleted", path)

                self._state = current_state
            except Exception as e:
                logger.error(f"PollingWatchObserver: Error in poll loop: {e}")

    def _trigger_callback(self, event_type: str, file_path: str):
        try:
            res = self.callback(event_type, file_path)
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)
        except Exception as e:
            logger.error(f"PollingWatchObserver: Callback error: {e}")

    def stop(self):
        """Stop the polling loop."""
        self.running = False
        if self._task:
            self._task.cancel()


class SemanticFS:
    """Enables semantic directory indexing and change triggers/monitoring."""

    def __init__(self, capability_manager: Optional[Any] = None):
        self._watchers: Dict[str, Tuple[str, Any]] = {}
        self._capability_manager = capability_manager
        # Logical VFS namespace -> physical root directory (backward compatible).
        self._vfs_roots: Dict[str, str] = {
            "skills": os.path.join("storage", "skills"),
            "sessions": os.path.join("storage", "sessions"),
            "research_memory": os.path.join("storage", "research_memory"),
            "memori": os.path.join("storage", "memori"),
            "output": "output",
            "temp": os.path.join("temp", "mcp_servers"),
        }
        self._vfs_subscribers: Dict[str, List[Callable[[str, str, str], Any]]] = {}

    def index_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Scan a directory and gather metadata of all contained files."""
        abs_path = os.path.abspath(directory_path)
        logger.info(f"SemanticFS: Indexing directory: {abs_path}")

        indexed_files = []
        if not os.path.exists(abs_path):
            logger.warning(f"SemanticFS: Path does not exist: {abs_path}")
            return []

        try:
            for root, _, files in os.walk(abs_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    try:
                        stat = os.stat(full_path)
                        _, ext = os.path.splitext(file)
                        indexed_files.append({
                            "name": file,
                            "path": full_path,
                            "size_bytes": stat.st_size,
                            "mtime": stat.st_mtime,
                            "extension": ext.lower(),
                        })
                    except OSError:
                        continue
        except Exception as e:
            logger.error(f"SemanticFS: Directory indexing failed: {e}")

        return indexed_files

    async def watch_directory(self, directory_path: str, callback: Callable[[str, str], Any]) -> str:
        """Watch a directory for file operations (created, modified, deleted)."""
        watch_id = f"watch_{uuid.uuid4().hex[:12]}"
        abs_path = os.path.abspath(directory_path)
        logger.info(f"SemanticFS: Watching directory: {abs_path} under id: {watch_id}")

        if WATCHDOG_AVAILABLE:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            try:
                class WatchdogHandler(FileSystemEventHandler):
                    def __init__(self, cb, loop):
                        self.cb = cb
                        self.loop = loop

                    def on_any_event(self, event):
                        if event.is_directory:
                            return
                        event_map = {
                            "created": "created",
                            "modified": "modified",
                            "deleted": "deleted",
                        }
                        if event.event_type in event_map:
                            try:
                                res = self.cb(event_map[event.event_type], event.src_path)
                                if asyncio.iscoroutine(res):
                                    if self.loop and self.loop.is_running():
                                        asyncio.run_coroutine_threadsafe(res, self.loop)
                                    else:
                                        res.close()
                            except Exception as err:
                                logger.error(f"SemanticFS (Watchdog): Callback error: {err}")

                event_handler = WatchdogHandler(callback, loop)
                observer = Observer()
                observer.schedule(event_handler, abs_path, recursive=True)
                observer.start()
                
                self._watchers[watch_id] = ("watchdog", observer)
                return watch_id
            except Exception as e:
                logger.warning(f"SemanticFS: Watchdog failed to initialize, falling back to polling: {e}")

        # Fallback to polling
        polling_observer = PollingWatchObserver(abs_path, callback)
        await polling_observer.start()
        self._watchers[watch_id] = ("polling", polling_observer)
        return watch_id

    def unwatch_directory(self, watch_id: str) -> bool:
        """Stop directory observation for a given watch ID."""
        if watch_id not in self._watchers:
            return False

        watcher_type, watcher_inst = self._watchers[watch_id]
        logger.info(f"SemanticFS: Stopping watch: {watch_id} ({watcher_type})")
        
        try:
            if watcher_type == "watchdog":
                watcher_inst.stop()
                watcher_inst.join()
            elif watcher_type == "polling":
                watcher_inst.stop()
            del self._watchers[watch_id]
            return True
        except Exception as e:
            logger.error(f"SemanticFS: Error while stopping watch {watch_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Unified Virtual Filesystem (VFS) layer
    # ------------------------------------------------------------------

    VFS_SCHEME = "vfs://"

    def register_vfs_root(self, namespace: str, physical_root: str) -> None:
        """Register or override a VFS namespace -> physical root mapping."""
        self._vfs_roots[namespace] = physical_root
        logger.info("SemanticFS: VFS root registered: %s -> %s", namespace, physical_root)

    def _parse_vfs_path(self, vfs_path: str) -> Tuple[str, str]:
        """Parse a ``vfs://<namespace>/<relative>`` path into (namespace, relative)."""
        if not vfs_path.startswith(self.VFS_SCHEME):
            raise ValueError(f"SemanticFS: Not a VFS path: {vfs_path}")
        remainder = vfs_path[len(self.VFS_SCHEME):]
        if "/" not in remainder:
            namespace, relative = remainder, ""
        else:
            namespace, relative = remainder.split("/", 1)
        if namespace not in self._vfs_roots:
            raise ValueError(f"SemanticFS: Unknown VFS namespace: {namespace}")
        return namespace, relative

    def _resolve_vfs_path(self, vfs_path: str) -> str:
        """Resolve a VFS logical path to its physical filesystem location."""
        namespace, relative = self._parse_vfs_path(vfs_path)
        root = self._vfs_roots[namespace]
        return os.path.join(root, relative) if relative else root

    def _check_capability(self, capability_name: str, agent_id: Optional[str] = None) -> bool:
        """Enforce capability-based access when a CapabilityManager is configured."""
        if self._capability_manager is None:
            return True
        if agent_id is None:
            return True
        try:
            return bool(self._capability_manager.agent_has(agent_id, capability_name))
        except Exception as exc:
            logger.error("SemanticFS: capability check failed for %s: %s", capability_name, exc)
            return False

    def read(self, vfs_path: str, agent_id: Optional[str] = None) -> bytes:
        """Read file contents via a VFS logical path."""
        if not self._check_capability("read_file", agent_id):
            raise PermissionError(f"SemanticFS: read_file capability denied for {agent_id}")
        physical = self._resolve_vfs_path(vfs_path)
        with open(physical, "rb") as handle:
            return handle.read()

    def write(self, vfs_path: str, data: bytes, agent_id: Optional[str] = None) -> str:
        """Write file contents via a VFS logical path, returning the physical path."""
        if not self._check_capability("write_file", agent_id):
            raise PermissionError(f"SemanticFS: write_file capability denied for {agent_id}")
        physical = self._resolve_vfs_path(vfs_path)
        os.makedirs(os.path.dirname(physical) or ".", exist_ok=True)
        with open(physical, "wb") as handle:
            handle.write(data)
        self._notify_vfs_subscribers("modified", vfs_path, physical)
        return physical

    def list(self, vfs_prefix: str, agent_id: Optional[str] = None) -> List[str]:
        """List VFS logical paths under a namespace or namespace/sub prefix."""
        if not self._check_capability("read_file", agent_id):
            raise PermissionError(f"SemanticFS: read_file capability denied for {agent_id}")
        namespace, relative = self._parse_vfs_path(vfs_prefix)
        root = self._vfs_roots[namespace]
        base = os.path.join(root, relative) if relative else root
        results: List[str] = []
        if not os.path.isdir(base):
            return results
        for current_root, _dirs, files in os.walk(base):
            for file_name in files:
                full = os.path.join(current_root, file_name)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                results.append(f"{self.VFS_SCHEME}{namespace}/{rel}")
        return results

    def delete(self, vfs_path: str, agent_id: Optional[str] = None) -> bool:
        """Delete a file via its VFS logical path."""
        if not self._check_capability("write_file", agent_id):
            raise PermissionError(f"SemanticFS: write_file capability denied for {agent_id}")
        physical = self._resolve_vfs_path(vfs_path)
        if not os.path.exists(physical):
            return False
        try:
            os.remove(physical)
            self._notify_vfs_subscribers("deleted", vfs_path, physical)
            return True
        except OSError as exc:
            logger.error("SemanticFS: delete failed for %s: %s", vfs_path, exc)
            return False

    def subscribe(self, vfs_prefix: str, callback: Callable[[str, str, str], Any]) -> str:
        """Subscribe to change events for VFS paths under ``vfs_prefix``.

        The callback receives ``(event_type, vfs_path, physical_path)``.
        Reuses the existing watch_directory implementation under the hood.
        """
        sub_id = f"vfs_sub_{uuid.uuid4().hex[:12]}"
        self._vfs_subscribers.setdefault(vfs_prefix, []).append(callback)

        namespace, relative = self._parse_vfs_path(vfs_prefix)
        root = self._vfs_roots[namespace]
        physical_dir = os.path.join(root, relative) if relative else root
        if not os.path.isdir(physical_dir):
            os.makedirs(physical_dir, exist_ok=True)

        def _bridge(event_type: str, physical_path: str) -> Any:
            try:
                rel = os.path.relpath(physical_path, root).replace(os.sep, "/")
                logical = f"{self.VFS_SCHEME}{namespace}/{rel}"
            except ValueError:
                return None
            return callback(event_type, logical, physical_path)

        # Schedule the async watcher without blocking the caller.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            asyncio.ensure_future(self.watch_directory(physical_dir, _bridge))
        else:
            loop.run_until_complete(self.watch_directory(physical_dir, _bridge))
        return sub_id

    def _notify_vfs_subscribers(self, event_type: str, vfs_path: str, physical_path: str) -> None:
        """Notify VFS subscribers of synchronous write/delete events."""
        for prefix, callbacks in list(self._vfs_subscribers.items()):
            if vfs_path.startswith(prefix):
                for callback in callbacks:
                    try:
                        res = callback(event_type, vfs_path, physical_path)
                        if asyncio.iscoroutine(res):
                            try:
                                loop = asyncio.get_event_loop()
                                asyncio.ensure_future(res)
                            except RuntimeError:
                                res.close()
                    except Exception as exc:
                        logger.error("SemanticFS: VFS subscriber error: %s", exc)
