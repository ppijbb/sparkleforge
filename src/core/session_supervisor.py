"""Anvil Phase O: crash detection + restart for the single-node session path.

AgentHarness.execute() already swallows and reports *expected* failures as
{"success": False, "error": ...} -- see issue #1506. What has no coverage is
an actual crash: an unhandled exception (or the process being killed mid-run)
that never returns at all. There is no systemd-style supervisor around the
CLI's single session invocation, so a crash just ends the process.

This wraps one session invocation in a restart loop. Restarts reuse the same
session_id so LangGraph's SQLite checkpointer (see langgraph_checkpointer.py)
can resume from the last completed graph node instead of starting over.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_RESTARTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 2.0
CRASH_EXHAUSTED_LABEL = "no-auto-fix"


def file_crash_exhausted_issue(session_id: str, exc: BaseException, max_restarts: int) -> str | None:
    """Anvil O-2: open a GitHub issue when a session crashes past the restart limit.

    Best-effort -- never raises. Labeled no-auto-fix per CLAUDE.md (a repeated
    crash needs a human to look at root cause, not an autofix loop retrying
    the same failure).
    """
    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not repo:
        logger.debug("[SessionSupervisor] GITHUB_REPOSITORY not set; skipping crash issue.")
        return None
    title = f"session {session_id} crashed {max_restarts} time(s) and gave up"
    body = (
        f"## Session crash exhausted restarts\n\n"
        f"Session `{session_id}` crashed {max_restarts} time(s) in a row "
        f"(Anvil Phase O session supervisor) and stopped retrying.\n\n"
        f"### Last exception\n\n```\n{type(exc).__name__}: {exc}\n```\n"
    )
    try:
        proc = subprocess.run(
            ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body,
             "--label", CRASH_EXHAUSTED_LABEL],
            text=True, capture_output=True, check=False, timeout=30,
        )
        if proc.returncode == 0:
            logger.info("[SessionSupervisor] filed crash issue: %s", proc.stdout.strip())
            return proc.stdout.strip()
        logger.warning("[SessionSupervisor] failed to file crash issue: %s", proc.stderr)
    except Exception:
        logger.warning("[SessionSupervisor] gh issue create unavailable", exc_info=True)
    return None


async def run_with_crash_supervision(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    session_id: str,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    on_exhausted: Callable[[str, BaseException], None] | None = None,
) -> T:
    """Run `coro_factory()`, restarting on an unhandled crash.

    KeyboardInterrupt/CancelledError are never swallowed -- those are the
    user or the runtime asking to stop, not a crash. Any other exception
    counts as a crash: log it, back off exponentially, and retry the same
    session up to `max_restarts` times. On final exhaustion, call
    `on_exhausted(session_id, exc)` (if given) and re-raise.
    """
    attempt = 0
    while True:
        try:
            if attempt:
                logger.warning(
                    "[SessionSupervisor] restarting session %s (attempt %d/%d)",
                    session_id, attempt, max_restarts,
                )
            return await coro_factory()
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            attempt += 1
            if attempt > max_restarts:
                logger.error(
                    "[SessionSupervisor] session %s crashed %d time(s), giving up: %s",
                    session_id, max_restarts, exc,
                )
                if on_exhausted is not None:
                    on_exhausted(session_id, exc)
                raise
            delay = backoff_base_seconds * (2 ** (attempt - 1))
            logger.warning(
                "[SessionSupervisor] session %s crashed (attempt %d/%d), "
                "retrying in %.1fs: %s",
                session_id, attempt, max_restarts, delay, exc,
            )
            await asyncio.sleep(delay)
