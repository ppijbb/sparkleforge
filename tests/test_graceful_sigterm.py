"""Issue #687: graceful SIGTERM shutdown for long-running CLI sessions.

SIGINT already surfaces as KeyboardInterrupt without extra wiring, but
SIGTERM (how Docker/systemd/process managers stop a service) previously
killed the process immediately, skipping scheduler.stop() and leaving
in-flight scheduled tasks in an incomplete state. These tests verify
_install_graceful_sigterm registers a handler that cancels the given task,
and degrades gracefully on platforms without signal-handler support.
"""

import asyncio
import os
import signal
from unittest.mock import MagicMock, patch

import main


def test_install_graceful_sigterm_registers_task_cancel_on_sigterm():
    mock_loop = MagicMock()
    task = MagicMock()

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        main._install_graceful_sigterm(task)

    mock_loop.add_signal_handler.assert_called_once_with(signal.SIGTERM, task.cancel)


def test_install_graceful_sigterm_swallows_not_implemented_error():
    mock_loop = MagicMock()
    mock_loop.add_signal_handler.side_effect = NotImplementedError("no signal support here")

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        main._install_graceful_sigterm(MagicMock())  # must not raise


def test_install_graceful_sigterm_swallows_runtime_error():
    mock_loop = MagicMock()
    mock_loop.add_signal_handler.side_effect = RuntimeError("no running event loop")

    with patch("asyncio.get_running_loop", return_value=mock_loop):
        main._install_graceful_sigterm(MagicMock())  # must not raise


def test_sigterm_handler_cancels_the_running_task():
    """End-to-end: a real SIGTERM to this process cancels the registered task.

    loop.add_signal_handler integrates with asyncio's own event loop (via a
    self-pipe), so raising the signal here is the standard, safe way asyncio
    signal handling is tested — it does not raise KeyboardInterrupt or crash
    the process the way an unhandled SIGTERM normally would.
    """

    async def run_test():
        cancelled = False

        async def long_running():
            await asyncio.sleep(10)

        task = asyncio.ensure_future(long_running())
        main._install_graceful_sigterm(task)

        os.kill(os.getpid(), signal.SIGTERM)

        try:
            await task
        except asyncio.CancelledError:
            cancelled = True

        return cancelled

    assert asyncio.run(run_test()) is True
