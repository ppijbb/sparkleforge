"""
ProcessManager - Centralized process lifecycle management

Provides:
- Single AbortController per run for coordinated cancellation
- Process registry to track all spawned child processes
- Signal propagation (SIGINT/SIGTERM) to all children
- Process tree killing where supported
- Terminal state restoration (cursor visibility)
- Configurable timeouts

Inspired by mdflow's process-manager.ts pattern.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TimeoutConfig:
    """Timeout configuration (in seconds)"""
    fetch_timeout: float = 10.0  # HTTP fetch operations
    command_timeout: float = 30.0  # Inline command execution
    agent_timeout: float = 0.0  # Agent execution (0 = no timeout)
    era_start_timeout: float = 30.0  # ERA server start timeout


@dataclass
class TrackedProcess:
    """Tracked process entry in the registry"""
    process: subprocess.Popen
    label: Optional[str] = None
    started_at: float = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now().timestamp()


class ProcessManager:
    """
    ProcessManager - Singleton for managing process lifecycle
    
    Usage:
        pm = ProcessManager.get_instance()
        pm.initialize()  # Sets up signal handlers
        
        # Register a process
        proc = subprocess.Popen([...])
        pm.register(proc, "my-command")
        
        # Process is automatically cleaned up on SIGINT/SIGTERM
        # Or manually unregister when done
        pm.unregister(proc)
    """
    
    _instance: Optional['ProcessManager'] = None
    
    def __init__(self):
        """Initialize ProcessManager (private, use get_instance())"""
        self.processes: Dict[int, TrackedProcess] = {}
        self.signal_handlers_installed = False
        self.cursor_hidden = False
        self._aborted = False
        
        # Timeout configuration from environment variables
        self.timeout_config = TimeoutConfig(
            fetch_timeout=float(os.getenv("SPARKLEFORGE_FETCH_TIMEOUT", "10.0")),
            command_timeout=float(os.getenv("SPARKLEFORGE_COMMAND_TIMEOUT", "30.0")),
            agent_timeout=float(os.getenv("SPARKLEFORGE_AGENT_TIMEOUT", "0.0")),
            era_start_timeout=float(os.getenv("SPARKLEFORGE_ERA_START_TIMEOUT", "30.0")),
        )
        
        # Cleanup callbacks
        self.cleanup_callbacks: List[Callable[[], Any]] = []
    
    @classmethod
    def get_instance(cls) -> 'ProcessManager':
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)"""
        if cls._instance:
            cls._instance.cleanup()
            cls._instance = None
    
    def initialize(self):
        """Initialize the ProcessManager and set up signal handlers"""
        if self.signal_handlers_installed:
            return
        
        def handle_signal(signum, frame):
            """Handle shutdown signal"""
            # 시그널 핸들러는 동기 함수이므로 이벤트 루프 확인 후 태스크 생성
            try:
                loop = asyncio.get_running_loop()
                # 중복 생성 방지
                if not hasattr(self, '_shutdown_task') or self._shutdown_task is None or self._shutdown_task.done():
                    def _schedule():
                        self._shutdown_task = asyncio.create_task(self._handle_shutdown(signum))
                    loop.call_soon_threadsafe(_schedule)
            except RuntimeError:
                # 이벤트 루프가 없으면 강제 종료
                self.abort()
                self.kill_all()
                self.restore_terminal()
                exit_code = 130 if signum == signal.SIGINT else 143
                os._exit(exit_code)
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        self.signal_handlers_installed = True
        logger.debug("ProcessManager signal handlers installed")
    
    @property
    def is_aborted(self) -> bool:
        """Check if the run has been aborted"""
        return self._aborted
    
    @property
    def timeouts(self) -> TimeoutConfig:
        """Get timeout configuration"""
        return self.timeout_config
    
    def set_timeouts(self, config: Dict[str, float]):
        """Update timeout configuration"""
        for key, value in config.items():
            if hasattr(self.timeout_config, key):
                setattr(self.timeout_config, key, value)
    
    def on_cleanup(self, callback: Callable[[], Any]):
        """Register a callback to run on cleanup/shutdown"""
        self.cleanup_callbacks.append(callback)
    
    def off_cleanup(self, callback: Callable[[], Any]):
        """Remove a cleanup callback"""
        if callback in self.cleanup_callbacks:
            self.cleanup_callbacks.remove(callback)
    
    def register(self, process: subprocess.Popen, label: Optional[str] = None):
        """Register a spawned process for tracking"""
        pid = process.pid
        tracked = TrackedProcess(process=process, label=label)
        self.processes[pid] = tracked
        
        logger.debug(f"Registered process {pid} ({label or 'unnamed'})")
        
        # Auto-unregister when process exits (async)
        async def wait_and_unregister():
            try:
                await asyncio.to_thread(process.wait)
            except Exception:
                pass
            finally:
                self.processes.pop(pid, None)
                logger.debug(f"Unregistered process {pid}")
        
        asyncio.create_task(wait_and_unregister())
    
    def unregister(self, process: subprocess.Popen):
        """Unregister a process from tracking"""
        pid = process.pid
        if pid in self.processes:
            del self.processes[pid]
            logger.debug(f"Unregistered process {pid}")
    
    def set_cursor_hidden(self, hidden: bool):
        """Mark cursor as hidden (called by spinner)"""
        self.cursor_hidden = hidden
    
    def restore_terminal(self):
        """Restore terminal state (cursor visibility)"""
        if self.cursor_hidden and sys.stderr.isatty():
            # Clear any partial output and show cursor
            sys.stderr.write("\r\x1B[K")  # Clear line
            sys.stderr.write("\x1B[?25h")  # Show cursor
            self.cursor_hidden = False
    
    def _kill_process(self, tracked: TrackedProcess) -> bool:
        """Kill a single process, attempting process group kill first"""
        proc = tracked.process
        label = tracked.label or "unnamed"
        
        try:
            # Try to kill the process group (negative PID) to kill all children
            # This only works on Unix-like systems and when the process is a group leader
            if os.name != 'nt':  # Not Windows
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    logger.debug(f"Killed process group for {proc.pid} ({label})")
                    return True
                except (OSError, ProcessLookupError):
                    # Process group kill failed (not a group leader or already dead)
                    pass
            
            # Fall back to direct kill
            proc.terminate()
            logger.debug(f"Terminated process {proc.pid} ({label})")
            return True
        except (ProcessLookupError, OSError) as e:
            # Process may have already exited
            logger.debug(f"Process {proc.pid} already exited: {e}")
            return False
    
    def kill_all(self) -> int:
        """Kill all registered processes"""
        killed = 0
        
        for pid, tracked in list(self.processes.items()):
            if self._kill_process(tracked):
                killed += 1
                
                # Wait for graceful termination (with timeout)
                try:
                    tracked.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    try:
                        tracked.process.kill()
                        tracked.process.wait()
                    except Exception:
                        pass
        
        # Clear the registry
        self.processes.clear()
        
        logger.info(f"Killed {killed} registered processes")
        return killed
    
    def abort(self):
        """Abort the current run"""
        if not self._aborted:
            self._aborted = True
            logger.debug("ProcessManager aborted")
    
    async def _handle_shutdown(self, signum: int):
        """Handle shutdown signal (SIGINT/SIGTERM)"""
        logger.info(f"Received signal {signum}, shutting down...")
        
        # Abort any pending operations
        self.abort()
        
        # Kill all child processes
        self.kill_all()
        
        # Restore terminal state
        self.restore_terminal()
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.debug(f"Cleanup callback error (ignored): {e}")
        
        # Exit with appropriate code
        # SIGINT = 2, SIGTERM = 15
        exit_code = 130 if signum == signal.SIGINT else 143
        sys.exit(exit_code)
    
    def cleanup(self):
        """Clean up resources (for testing)"""
        self.kill_all()
        self.restore_terminal()
        self._aborted = False
        self.cleanup_callbacks.clear()
        self.timeout_config = TimeoutConfig()
    
    @property
    def active_count(self) -> int:
        """Get the number of active processes"""
        return len(self.processes)
    
    def get_active_processes(self) -> List[Dict[str, Any]]:
        """Get info about active processes (for debugging)"""
        now = datetime.now().timestamp()
        return [
            {
                "pid": pid,
                "label": tracked.label,
                "running_seconds": now - tracked.started_at,
            }
            for pid, tracked in self.processes.items()
        ]


def get_process_manager() -> ProcessManager:
    """Convenience function to get the ProcessManager instance"""
    return ProcessManager.get_instance()


async def with_timeout(
    promise: asyncio.Task | asyncio.Future,
    timeout_seconds: float,
    message: str = "Operation timed out"
) -> Any:
    """
    Race a promise against a timeout
    Returns the promise result or raises TimeoutError on timeout/abort
    """
    if timeout_seconds <= 0:
        return await promise
    
    pm = get_process_manager()
    
    try:
        return await asyncio.wait_for(promise, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        if pm.is_aborted:
            raise asyncio.CancelledError("Operation cancelled")
        raise TimeoutError(f"{message} (timeout: {timeout_seconds}s)")
    except asyncio.CancelledError:
        if pm.is_aborted:
            raise asyncio.CancelledError("Operation cancelled")
        raise

