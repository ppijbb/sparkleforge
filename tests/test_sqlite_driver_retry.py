"""Issue #676: SQLite lock-contention retries.

Verifies the busy-retry helper in src/core/db/sqlite_driver.py retries on
"database is locked"/"database is busy" errors with backoff, but does not
retry (or mask) other OperationalErrors, and that a real SQLiteDriver
survives simulated lock contention end-to-end.
"""

import sqlite3

import pytest

from src.core.db.sqlite_driver import (
    SQLiteDriver,
    _execute_with_busy_retry,
    _is_busy_error,
)


def test_is_busy_error_matches_locked_and_busy():
    assert _is_busy_error(sqlite3.OperationalError("database is locked"))
    assert _is_busy_error(sqlite3.OperationalError("database is busy"))


def test_is_busy_error_rejects_other_operational_errors():
    assert not _is_busy_error(sqlite3.OperationalError("no such table: foo"))
    assert not _is_busy_error(ValueError("database is locked"))


@pytest.mark.asyncio
async def test_execute_with_busy_retry_succeeds_after_transient_lock():
    attempts = {"count": 0}

    async def flaky_op():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    result = await _execute_with_busy_retry(flaky_op)
    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_execute_with_busy_retry_does_not_retry_other_errors():
    attempts = {"count": 0}

    async def broken_op():
        attempts["count"] += 1
        raise sqlite3.OperationalError("no such table: foo")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        await _execute_with_busy_retry(broken_op)
    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_execute_with_busy_retry_raises_after_exhausting_attempts():
    async def always_locked():
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        await _execute_with_busy_retry(always_locked)


@pytest.mark.asyncio
async def test_sqlite_driver_execute_survives_transient_lock(tmp_path, monkeypatch):
    driver = SQLiteDriver(str(tmp_path / "retry_test.db"))
    await driver.connect()
    await driver.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")

    real_execute = driver._connection.execute
    call_count = {"n": 0}

    async def flaky_execute(query, *args, **kwargs):
        if "INSERT" in query and call_count["n"] == 0:
            call_count["n"] += 1
            raise sqlite3.OperationalError("database is locked")
        return await real_execute(query, *args, **kwargs)

    monkeypatch.setattr(driver._connection, "execute", flaky_execute)

    await driver.execute("INSERT INTO t (val) VALUES ('hello')")
    row = await driver.fetch_one("SELECT val FROM t WHERE id = 1")

    assert call_count["n"] == 1
    assert row["val"] == "hello"

    await driver.disconnect()
