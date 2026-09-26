"""Unit and integration tests for SDK asynchronous job API (Issue #1622)."""

import asyncio
import json
from typing import Any, Dict
import pytest

from src import sdk
from src.sdk import JobStatus, get_job_status, get_report, submit_job


@pytest.fixture(autouse=True)
def _clear_jobs(tmp_path, monkeypatch):
    # #1654: isolate the durable job-store file per test -- never touch the
    # real ~/.sparkleforge/sdk_jobs.json, and never leak state between tests.
    monkeypatch.setattr(sdk, "_JOB_STORE_PATH", tmp_path / "sdk_jobs.json")
    sdk._jobs.clear()
    sdk._job_tasks.clear()
    yield
    sdk._jobs.clear()
    sdk._job_tasks.clear()


@pytest.mark.asyncio
async def test_job_status_dataclass_methods():
    status = JobStatus(
        job_id="test-123",
        status="completed",
        topic="AI Safety",
        submitted_at="2026-09-16T12:00:00Z",
        completed_at="2026-09-16T12:01:00Z",
        error=None,
        result={"summary": "All good"},
        user_id="user-456",
    )

    d = status.to_dict()
    assert d["job_id"] == "test-123"
    assert d["status"] == "completed"
    assert d["topic"] == "AI Safety"
    assert d["result"] == {"summary": "All good"}
    assert d["user_id"] == "user-456"

    # Dict-like access
    assert status["status"] == "completed"
    assert status["topic"] == "AI Safety"
    assert status.get("job_id") == "test-123"
    assert status.get("missing_key", "default") == "default"


@pytest.mark.asyncio
async def test_submit_job_and_poll_success(monkeypatch):
    async def fake_run(prompt: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"report": f"findings for {prompt}", "score": 0.95}

    monkeypatch.setattr(sdk, "run", fake_run)

    job_id = await submit_job("quantum supremacy")
    assert isinstance(job_id, str)
    assert len(job_id) > 0

    # Polling until completed
    for _ in range(50):
        status = await get_job_status(job_id)
        if status.status == "completed":
            break
        await asyncio.sleep(0.01)

    assert status.status == "completed"
    assert status.topic == "quantum supremacy"
    assert status.error is None
    assert status.completed_at is not None

    report = await get_report(job_id)
    assert report is not None
    assert report["report"] == "findings for quantum supremacy"
    assert report["score"] == 0.95


@pytest.mark.asyncio
async def test_submit_job_handles_failure(monkeypatch):
    async def failing_run(prompt: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        raise RuntimeError("LLM synthesis timeout")

    monkeypatch.setattr(sdk, "run", failing_run)

    job_id = await submit_job("failing prompt")

    for _ in range(50):
        status = await get_job_status(job_id)
        if status.status == "failed":
            break
        await asyncio.sleep(0.01)

    assert status.status == "failed"
    assert "LLM synthesis timeout" in (status.error or "")

    report = await get_report(job_id)
    assert report is None


@pytest.mark.asyncio
async def test_get_job_status_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Job not found"):
        await get_job_status("non-existent-uuid")


@pytest.mark.asyncio
async def test_get_report_unknown_returns_none():
    report = await get_report("non-existent-uuid")
    assert report is None


@pytest.mark.asyncio
async def test_submit_job_with_supabase_integration(monkeypatch):
    from src.utils import supabase_exporter

    created_job_id = "sb-job-999"
    updates = []

    monkeypatch.setattr(supabase_exporter, "get_supabase_client", lambda: object())

    async def fake_create_job(topic: str, user_id: str | None = None):
        return {"id": created_job_id, "topic": topic, "user_id": user_id}

    async def fake_update_job_status(job_id: str, status: str, error_message: str | None = None):
        updates.append((job_id, status, error_message))
        return True

    async def fake_publish_report(self, report: dict):
        pass

    async def fake_run(prompt: str) -> Dict[str, Any]:
        return {"title": "supabase test report"}

    monkeypatch.setattr(supabase_exporter, "create_job", fake_create_job)
    monkeypatch.setattr(supabase_exporter, "update_job_status", fake_update_job_status)
    monkeypatch.setattr(supabase_exporter.SupabaseExporter, "publish_report", fake_publish_report)
    monkeypatch.setattr(sdk, "run", fake_run)

    job_id = await submit_job("remote topic", user_id="usr-1")
    assert job_id == created_job_id

    for _ in range(50):
        status = await get_job_status(job_id)
        if status.status == "completed":
            break
        await asyncio.sleep(0.01)

    assert status.status == "completed"
    # Ensure updates were sent to Supabase
    statuses = [u[1] for u in updates]
    assert "running" in statuses
    assert "completed" in statuses


@pytest.mark.asyncio
async def test_completed_job_survives_simulated_process_restart(monkeypatch):
    async def fake_run(prompt: str) -> Dict[str, Any]:
        return {"report": f"findings for {prompt}"}

    monkeypatch.setattr(sdk, "run", fake_run)

    job_id = await submit_job("durable topic")
    for _ in range(50):
        status = await get_job_status(job_id)
        if status.status == "completed":
            break
        await asyncio.sleep(0.01)
    assert status.status == "completed"

    # Simulate a process restart: a fresh process has empty _jobs/_job_tasks,
    # but the same job-store file on disk.
    sdk._jobs.clear()
    sdk._job_tasks.clear()

    status_after_restart = await get_job_status(job_id)
    assert status_after_restart.status == "completed"
    assert status_after_restart.topic == "durable topic"

    report_after_restart = await get_report(job_id)
    assert report_after_restart == {"report": "findings for durable topic"}


@pytest.mark.asyncio
async def test_orphaned_running_job_reports_failed_after_restart(monkeypatch):
    # Never resolves on its own -- simulates a job whose task died with the process.
    stuck = asyncio.Event()

    async def hanging_run(prompt: str) -> Dict[str, Any]:
        await stuck.wait()
        return {}

    monkeypatch.setattr(sdk, "run", hanging_run)

    job_id = await submit_job("orphaned topic")
    for _ in range(50):
        status = await get_job_status(job_id)
        if status.status == "running":
            break
        await asyncio.sleep(0.01)
    assert status.status == "running"

    # Simulate a process restart while the job was still in flight.
    sdk._job_tasks[job_id].cancel()
    sdk._jobs.clear()
    sdk._job_tasks.clear()

    status_after_restart = await get_job_status(job_id)
    assert status_after_restart.status == "failed"
    assert "restarted" in (status_after_restart.error or "")


def test_persist_job_is_a_noop_for_unknown_job_id():
    sdk._persist_job("no-such-job")  # must not raise, and must not create the file
    assert not sdk._JOB_STORE_PATH.exists()


def test_load_job_store_fails_open_on_corrupt_file():
    sdk._JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sdk._JOB_STORE_PATH.write_text("not valid json", encoding="utf-8")

    assert sdk._load_job_store() == {}


def test_prune_expired_jobs_drops_old_terminal_jobs_only():
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(days=sdk._JOB_STORE_TTL_DAYS + 1)).isoformat()
    recent = datetime.now(timezone.utc).isoformat()
    store = {
        "old-completed": {"status": "completed", "completed_at": old},
        "old-failed": {"status": "failed", "completed_at": old},
        "recent-completed": {"status": "completed", "completed_at": recent},
        "still-running": {"status": "running", "completed_at": None},
        "no-completed-at": {"status": "completed"},
    }

    pruned = sdk._prune_expired_jobs(store)

    assert set(pruned) == {"recent-completed", "still-running", "no-completed-at"}


def test_persist_job_prunes_old_terminal_jobs_from_the_store(monkeypatch):
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(days=sdk._JOB_STORE_TTL_DAYS + 1)).isoformat()
    sdk._JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sdk._JOB_STORE_PATH.write_text(
        json.dumps({"ancient-job": {"status": "completed", "completed_at": old}}),
        encoding="utf-8",
    )

    sdk._jobs["new-job"] = {"job_id": "new-job", "status": "running"}
    sdk._persist_job("new-job")

    on_disk = json.loads(sdk._JOB_STORE_PATH.read_text(encoding="utf-8"))
    assert set(on_disk) == {"new-job"}


@pytest.mark.asyncio
async def test_get_job_status_and_report_fallback_from_supabase(monkeypatch):
    from src.utils import supabase_exporter

    monkeypatch.setattr(supabase_exporter, "get_supabase_client", lambda: object())

    remote_id = "remote-uuid-111"

    async def fake_sb_get_job_status(job_id: str):
        if job_id == remote_id:
            return {
                "id": remote_id,
                "topic": "remote topic",
                "status": "completed",
                "created_at": "2026-09-16T10:00:00Z",
                "updated_at": "2026-09-16T10:05:00Z",
                "user_id": "usr-remote",
            }
        return None

    async def fake_sb_get_report(report_id: str):
        if report_id == remote_id:
            return {"id": remote_id, "summary": "remote report content"}
        return None

    monkeypatch.setattr(supabase_exporter, "get_job_status", fake_sb_get_job_status)
    monkeypatch.setattr(supabase_exporter, "get_report", fake_sb_get_report)

    # Note: remote_id is NOT in sdk._jobs
    assert remote_id not in sdk._jobs

    status = await get_job_status(remote_id)
    assert status.job_id == remote_id
    assert status.status == "completed"
    assert status.topic == "remote topic"
    assert status.user_id == "usr-remote"

    report = await get_report(remote_id)
    assert report is not None
    assert report["summary"] == "remote report content"
