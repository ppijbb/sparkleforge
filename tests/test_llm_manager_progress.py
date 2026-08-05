"""with_progress() must always return the wrapped coroutine's result,
whether or not it decides to print a ticker."""

import asyncio
from unittest.mock import patch

from src.core.llm_manager.progress import with_progress


async def _quick_result():
    await asyncio.sleep(0.01)
    return "ok"


def test_returns_result_when_not_a_tty():
    with patch("sys.stdout.isatty", return_value=False):
        result = asyncio.run(with_progress(_quick_result(), label="test"))
    assert result == "ok"


def test_returns_result_and_ticks_when_a_tty():
    async def _slower_result():
        await asyncio.sleep(1.2)
        return "ok"

    with patch("sys.stdout.isatty", return_value=True):
        result = asyncio.run(with_progress(_slower_result(), label="test"))
    assert result == "ok"


def test_propagates_exceptions():
    async def _fails():
        await asyncio.sleep(0.01)
        raise ValueError("boom")

    with patch("sys.stdout.isatty", return_value=False):
        try:
            asyncio.run(with_progress(_fails(), label="test"))
            assert False, "expected ValueError to propagate"
        except ValueError as e:
            assert str(e) == "boom"


if __name__ == "__main__":
    test_returns_result_when_not_a_tty()
    test_returns_result_and_ticks_when_a_tty()
    test_propagates_exceptions()
    print("ok")
