"""Anvil Phase O: session crash detection, restart, backoff limit, and issue-on-exhaustion."""

import asyncio

import pytest

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
