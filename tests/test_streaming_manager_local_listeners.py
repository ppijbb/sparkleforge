"""#1621: StreamingManager's in-process local-listener registry.

src.sdk.run()'s on_progress relies on stream_event() notifying plain async
callbacks in addition to WebSocket connections -- these tests cover that
registry directly (add/remove, notification-on-stream_event, and that a
raising listener never breaks stream_event for the caller or other listeners).
"""

import asyncio

from src.core.streaming_manager import EventType, StreamingManager


def test_add_local_listener_receives_streamed_events():
    manager = StreamingManager()
    received = []

    async def listener(event):
        received.append(event)

    manager.add_local_listener(listener)

    asyncio.run(
        manager.stream_event(
            event_type=EventType.WORKFLOW_START,
            agent_id="orchestrator",
            workflow_id="wf1",
            data={"message": "starting"},
        )
    )

    assert len(received) == 1
    assert received[0].workflow_id == "wf1"
    assert received[0].data["message"] == "starting"


def test_remove_local_listener_stops_notifications():
    manager = StreamingManager()
    received = []

    async def listener(event):
        received.append(event)

    manager.add_local_listener(listener)
    manager.remove_local_listener(listener)

    asyncio.run(
        manager.stream_event(
            event_type=EventType.WORKFLOW_START,
            agent_id="orchestrator",
            workflow_id="wf1",
            data={},
        )
    )

    assert received == []


def test_remove_local_listener_is_a_noop_if_never_added():
    manager = StreamingManager()

    async def listener(event):
        pass

    manager.remove_local_listener(listener)  # must not raise


def test_raising_local_listener_does_not_break_stream_event_or_other_listeners():
    manager = StreamingManager()
    received = []

    async def bad_listener(event):
        raise RuntimeError("boom")

    async def good_listener(event):
        received.append(event)

    manager.add_local_listener(bad_listener)
    manager.add_local_listener(good_listener)

    ok = asyncio.run(
        manager.stream_event(
            event_type=EventType.WORKFLOW_START,
            agent_id="orchestrator",
            workflow_id="wf1",
            data={},
        )
    )

    assert ok is True
    assert len(received) == 1
