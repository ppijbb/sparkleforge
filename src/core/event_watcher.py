import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class EventWatcher:
    """Event-driven sleep/wake architecture.
    Sleeps continuously and polls via a provided function.
    Wakes up and delegates to a handler when an event is detected.
    """

    def __init__(
        self,
        poll_fn: Callable[[], Awaitable[Any]],
        handle_fn: Callable[[Any], Awaitable[None]],
        interval: int = 5,
    ):
        """:param poll_fn: An async function that checks for events. Should return None if no event.
        :param handle_fn: An async function that processes the event payload.
        :param interval: Sleep interval in seconds.
        """
        self.poll_fn = poll_fn
        self.handle_fn = handle_fn
        self.interval = interval
        self._running = False
        self._task = None

    async def start(self):
        """Start the event polling loop."""
        if self._running:
            logger.warning("EventWatcher is already running.")
            return

        self._running = True
        logger.info(f"EventWatcher started. Polling every {self.interval} seconds.")

        while self._running:
            try:
                # Polling for events
                event_data = await self.poll_fn()

                if event_data:
                    logger.info("🚨 EventWatcher: Event detected! Waking up...")
                    try:
                        await self.handle_fn(event_data)
                    except Exception as e:
                        logger.error(f"EventWatcher: Error handling event: {e}")
                    logger.info("💤 EventWatcher: Handler completed. Returning to sleep...")
            except Exception as e:
                logger.error(f"EventWatcher: Error during polling: {e}")

            # Sleep until next poll to save resources
            if self._running:
                for _ in range(self.interval):
                    if not self._running:
                        break
                    await asyncio.sleep(1)

    def stop(self):
        """Stop the event polling loop."""
        self._running = False
        logger.info("EventWatcher stopped.")
