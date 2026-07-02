import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Dict, List

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

    def __init__(self):
        self._watchers = {}

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
