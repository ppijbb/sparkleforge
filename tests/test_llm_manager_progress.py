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


def test_returns_result_when_stdout_has_no_isatty():
    """sys.stdout can be swapped for a wrapper without isatty() at all,
    e.g. SupabaseStdoutRedirector during a research run -- must not crash."""

    class _NoIsattyStdout:
        def write(self, text):
            pass

        def flush(self):
            pass

    with patch("sys.stdout", _NoIsattyStdout()):
        result = asyncio.run(with_progress(_quick_result(), label="test"))
    assert result == "ok"


def test_returns_result_and_ticks_when_a_tty(capsys):
    # with_progress() now renders through rich's shared get_console() singleton
    # (so it cooperates with an active spinner/Live elsewhere in the process
    # instead of fighting it for the terminal -- two independent Console
    # instances can't coordinate a shared Live region). That singleton binds
    # to whatever sys.stdout was at its first-ever use in this process, which
    # predates capsys's per-test stdout swap -- so this test needs its own
    # fresh Console bound to the *current* (captured) stdout, or it would
    # write to a stream capsys.readouterr() here can no longer see.
    #
    # output_manager must use that same shared console object even when
    # colors are disabled, so a private no-color Console never competes with
    # the singleton spinner (see issue #1375).
    from rich.console import Console

    async def _slower_result():
        await asyncio.sleep(0.05)
        return "ok"

    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("src.core.llm_manager.progress.get_console", return_value=Console()),
    ):
        result = asyncio.run(
            with_progress(_slower_result(), label="test", interval=0.01)
        )
    assert result == "ok"
    assert "test..." in capsys.readouterr().out


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
