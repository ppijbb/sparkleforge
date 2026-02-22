"""Session lane: per-session queue and run serialization (OpenClaw-style).

One run per session at a time; other sessions run in parallel.
Integrates with task_monitoring for queue depth, latency, and success/failure.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Tuple

from src.core.input_router import InputEnvelope
from src.core.task_monitoring import get_task_metrics_collector

logger = logging.getLogger(__name__)

# (envelope, future) so worker can set result
_Item = Tuple[InputEnvelope, asyncio.Future[Any]]


class SessionLane:
    """Per-session queue with a single active run per session."""

    def __init__(
        self,
        run_fn: Callable[[str, InputEnvelope], Awaitable[Any]] | None = None,
    ) -> None:
        self._run_fn = run_fn
        self._queues: Dict[str, asyncio.Queue[_Item]] = {}
        self._workers: Dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def set_run_fn(self, run_fn: Callable[[str, InputEnvelope], Awaitable[Any]]) -> None:
        """Set the function to run for each envelope (e.g. orchestrator.execute)."""
        self._run_fn = run_fn

    async def _get_queue(self, session_id: str) -> asyncio.Queue[_Item]:
        async with self._lock:
            if session_id not in self._queues:
                self._queues[session_id] = asyncio.Queue()
            return self._queues[session_id]

    async def _worker(self, session_id: str) -> None:
        queue = await self._get_queue(session_id)
        run_fn = self._run_fn
        if not run_fn:
            logger.warning("SessionLane worker started with no run_fn")
            return
        collector = get_task_metrics_collector()
        while True:
            try:
                item: _Item = await queue.get()
                try:
                    envelope, future = item
                    try:
                        collector.record_dequeued(session_id)
                        collector.record_start(session_id, task_id=None)
                    except Exception:
                        pass
                    start = time.monotonic()
                    try:
                        result = await run_fn(session_id, envelope)
                        if not future.done():
                            future.set_result(result)
                        try:
                            collector.record_end(
                                session_id, success=True, latency_sec=time.monotonic() - start
                            )
                        except Exception:
                            pass
                    except asyncio.CancelledError:
                        try:
                            collector.record_end(
                                session_id, success=False, latency_sec=time.monotonic() - start
                            )
                        except Exception:
                            pass
                        if not future.done():
                            future.cancel()
                        raise
                    except Exception as e:
                        logger.exception("SessionLane run failed: %s", e)
                        try:
                            collector.record_end(
                                session_id, success=False, latency_sec=time.monotonic() - start
                            )
                        except Exception:
                            pass
                        if not future.done():
                            future.set_exception(e)
                finally:
                    queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("SessionLane worker error: %s", e)

    async def _ensure_worker(self, session_id: str) -> None:
        async with self._lock:
            if session_id in self._workers and not self._workers[session_id].done():
                return
            self._workers[session_id] = asyncio.create_task(self._worker(session_id))

    async def enqueue_and_wait(
        self,
        session_id: str,
        envelope: InputEnvelope,
        run_fn: Callable[[str, InputEnvelope], Awaitable[Any]] | None = None,
    ) -> Any:
        """Enqueue envelope for session and wait for result. One run per session at a time."""
        fn = run_fn or self._run_fn
        if not fn:
            raise RuntimeError("SessionLane: run_fn not set and not passed to enqueue_and_wait")
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        queue = await self._get_queue(session_id)
        try:
            get_task_metrics_collector().record_enqueued(
                session_id, getattr(envelope, "type", "message")
            )
        except Exception:
            pass
        await queue.put((envelope, future))
        await self._ensure_worker(session_id)
        return await future

    def enqueue(
        self,
        session_id: str,
        envelope: InputEnvelope,
    ) -> asyncio.Future[Any]:
        """Enqueue and return a Future for the result. Caller must ensure worker is started."""
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        asyncio.create_task(self._put_and_ensure(session_id, envelope, future))
        return future

    async def _put_and_ensure(
        self,
        session_id: str,
        envelope: InputEnvelope,
        future: asyncio.Future[Any],
    ) -> None:
        try:
            get_task_metrics_collector().record_enqueued(
                session_id, getattr(envelope, "type", "message")
            )
        except Exception:
            pass
        queue = await self._get_queue(session_id)
        await queue.put((envelope, future))
        await self._ensure_worker(session_id)


# Optional global instance for app-wide session lane
_session_lane: SessionLane | None = None


def get_session_lane() -> SessionLane:
    """Return global SessionLane instance."""
    global _session_lane
    if _session_lane is None:
        _session_lane = SessionLane()
    return _session_lane


def make_cron_execution_callback(orchestrator: Any) -> Callable[[str, str], Awaitable[Any]]:
    """Return an async (user_query, session_id) callback for scheduler using cron envelope + lane."""
    from src.core.input_router import normalize_cron

    async def callback(user_query: str, session_id: str) -> Any:
        envelope = normalize_cron(session_id, user_query)
        return await orchestrator.execute_via_lane(session_id, envelope)

    return callback


async def start_heartbeat_loop(
    interval_seconds: float,
    session_id: str = "heartbeat",
    lane: SessionLane | None = None,
) -> None:
    """Emit heartbeat envelopes every interval_seconds. Lane must have run_fn set (e.g. via set_run_fn).
    Heartbeat is typically suppressed in run_fn unless there is pending work."""
    from src.core.input_router import normalize_heartbeat

    lane = lane or get_session_lane()
    logger.info("Heartbeat loop started (interval=%s s, session_id=%s)", interval_seconds, session_id)
    while True:
        await asyncio.sleep(interval_seconds)
        envelope = normalize_heartbeat(session_id)
        try:
            lane.enqueue(session_id, envelope)
        except Exception as e:
            logger.warning("Heartbeat enqueue failed: %s", e)
