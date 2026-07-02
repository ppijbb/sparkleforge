import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)


class EventBus:
    """Lightweight asynchronous event bus for observation telemetry."""

    def __init__(self):
        self._listeners: Dict[str, Dict[str, Callable[[Any], Awaitable[None]]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Any], Awaitable[None]]) -> str:
        """Subscribe to an event type. Returns a unique subscription ID."""
        sub_id = f"sub_{uuid.uuid4().hex[:12]}"
        if event_type not in self._listeners:
            self._listeners[event_type] = {}
        self._listeners[event_type][sub_id] = callback
        logger.debug(f"EventBus: Subscribed listener {sub_id} to event '{event_type}'")
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Cancel a subscription using its ID."""
        for event_type, subs in list(self._listeners.items()):
            if subscription_id in subs:
                del subs[subscription_id]
                logger.debug(f"EventBus: Unsubscribed listener {subscription_id} from '{event_type}'")
                if not subs:
                    del self._listeners[event_type]
                return True
        return False

    async def publish(self, event_type: str, data: Any):
        """Asynchronously publish an event to all subscribers."""
        if event_type not in self._listeners:
            return

        listeners = list(self._listeners[event_type].values())
        if not listeners:
            return

        tasks = []
        for listener in listeners:
            try:
                res = listener(data)
                if asyncio.iscoroutine(res):
                    tasks.append(asyncio.create_task(res))
            except Exception as e:
                logger.error(f"EventBus: Error executing or scheduling listener callback for '{event_type}': {e}")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.error(f"EventBus: Error in listener execution for '{event_type}': {res}")
