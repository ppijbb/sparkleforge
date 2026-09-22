"""Anvil Phase O: session crash detection, restart, backoff limit, and issue-on-exhaustion."""

import asyncio

import pytest

import src.core.session_supervisor as session_supervisor
from src.core.session_supervisor import run_with_crash_supervision


@pytest.mark.asyncio
async def test_succeeds_without_restart():
    calls = []

    async def factory():
        calls.append(1)
        return "ok"

    result = await run_with_crash_supervision(factory, session_id="s1", backoff_base_seconds=0)
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_restarts_same_session_after_crash_then_succeeds():
    attempts = []

    async def factory():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("simulated crash")
        return "recovered"

    result = await run_with_crash_supervision(
        factory, session_id="s2", max_restarts=5, backoff_base_seconds=0
    )
    assert result == "recovered"
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_gives_up_and_calls_on_exhausted_after_max_restarts():
    exhausted = []

    async def factory():
        raise RuntimeError("always crashes")

    with pytest.raises(RuntimeError):
        await run_with_crash_supervision(
            factory,
            session_id="s3",
            max_restarts=2,
            backoff_base_seconds=0,
            on_exhausted=lambda sid, exc: exhausted.append((sid, str(exc))),
        )

    assert exhausted == [("s3", "always crashes")]


@pytest.mark.asyncio
async def test_crash_and_restart_are_logged_to_history(monkeypatch):
    crash_events = []
    monkeypatch.setattr(
        session_supervisor,
        "log_history_event",
        lambda session_id, event_type, content, **kw: crash_events.append(
            (session_id, event_type, content, kw)
        ),
    )
    ended = []
    monkeypatch.setattr(
        session_supervisor,
        "end_history_session",
        lambda session_id, **kw: ended.append((session_id, kw)),
    )

    attempts = []

    async def factory():
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("simulated crash")
        return "recovered"

    result = await run_with_crash_supervision(
        factory, session_id="s5", max_restarts=5, backoff_base_seconds=0
    )

    assert result == "recovered"
    assert len(crash_events) == 1
    session_id, event_type, content, kw = crash_events[0]
    assert session_id == "s5"
    assert event_type == "crash"
    assert content == "simulated crash"
    assert kw["metadata"]["restart_count"] == 1
    assert kw["metadata"]["restart_attempt"] == 1
    assert ended == []


@pytest.mark.asyncio
async def test_exhausted_restarts_ends_history_session_as_failed(monkeypatch):
    monkeypatch.setattr(session_supervisor, "log_history_event", lambda *a, **kw: None)
    ended = []
    monkeypatch.setattr(
        session_supervisor,
        "end_history_session",
        lambda session_id, **kw: ended.append((session_id, kw)),
    )

    async def factory():
        raise RuntimeError("always crashes")

    with pytest.raises(RuntimeError):
        await run_with_crash_supervision(
            factory, session_id="s6", max_restarts=2, backoff_base_seconds=0
        )

    assert len(ended) == 1
    session_id, kw = ended[0]
    assert session_id == "s6"
    assert kw["status"] == "failed"
    assert kw["metadata"]["reason"] == "crash_limit_exceeded"
    assert kw["metadata"]["restart_count"] == 2


@pytest.mark.asyncio
async def test_keyboard_interrupt_is_not_treated_as_a_crash():
    async def factory():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        await run_with_crash_supervision(factory, session_id="s4", backoff_base_seconds=0)


if __name__ == "__main__":
    asyncio.run(test_succeeds_without_restart())
    asyncio.run(test_restarts_same_session_after_crash_then_succeeds())
    asyncio.run(test_gives_up_and_calls_on_exhausted_after_max_restarts())
    print("ok")
