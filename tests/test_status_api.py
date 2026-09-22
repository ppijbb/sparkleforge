"""Regression tests for the read-only status/report API (src/web/status_api.py)."""

import time
from typing import Any, Dict

import pytest
from starlette.testclient import TestClient

from src import sdk
from src.web import status_api

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _token(monkeypatch, tmp_path):
    monkeypatch.setenv("STATUS_API_TOKEN", "test-token")
    # #1654: isolate the durable job-store file per test.
    monkeypatch.setattr(sdk, "_JOB_STORE_PATH", tmp_path / "sdk_jobs.json")
    sdk._jobs.clear()
    sdk._job_tasks.clear()
    yield
    sdk._jobs.clear()
    sdk._job_tasks.clear()


def test_job_status_401_without_token():
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status")

    assert response.status_code == 401


def test_job_status_401_with_wrong_token():
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_job_status_503_when_token_not_configured(monkeypatch):
    monkeypatch.delenv("STATUS_API_TOKEN", raising=False)
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers=AUTH)

    assert response.status_code == 503


def test_job_status_503_when_supabase_unconfigured(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: None)
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers=AUTH)

    assert response.status_code == 503


def test_job_status_404_when_job_missing(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_job_status(job_id):
        return None

    monkeypatch.setattr(status_api, "get_job_status", fake_get_job_status)
    client = TestClient(status_api.app)

    response = client.get("/jobs/missing-id/status", headers=AUTH)

    assert response.status_code == 404


def test_job_status_503_when_query_fails(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_job_status(job_id):
        raise status_api.SupabaseQueryError("network down")

    monkeypatch.setattr(status_api, "get_job_status", fake_get_job_status)
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers=AUTH)

    assert response.status_code == 503


def test_job_status_200_returns_job_row(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_job_status(job_id):
        return {"id": job_id, "status": "running", "topic": "quantum ML"}

    monkeypatch.setattr(status_api, "get_job_status", fake_get_job_status)
    client = TestClient(status_api.app)

    response = client.get("/jobs/abc/status", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"id": "abc", "status": "running", "topic": "quantum ML"}


def test_report_401_without_token():
    client = TestClient(status_api.app)

    response = client.get("/reports/xyz")

    assert response.status_code == 401


def test_report_404_when_missing(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_report(report_id):
        return None

    monkeypatch.setattr(status_api, "get_report", fake_get_report)
    client = TestClient(status_api.app)

    response = client.get("/reports/missing-id", headers=AUTH)

    assert response.status_code == 404


def test_report_503_when_query_fails(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_report(report_id):
        raise status_api.SupabaseQueryError("network down")

    monkeypatch.setattr(status_api, "get_report", fake_get_report)
    client = TestClient(status_api.app)

    response = client.get("/reports/xyz", headers=AUTH)

    assert response.status_code == 503


def test_report_200_returns_report_row(monkeypatch):
    monkeypatch.setattr(status_api, "get_supabase_client", lambda: object())

    async def fake_get_report(report_id):
        return {"id": report_id, "topic": "quantum ML", "full_report": "..."}

    monkeypatch.setattr(status_api, "get_report", fake_get_report)
    client = TestClient(status_api.app)

    response = client.get("/reports/xyz", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["id"] == "xyz"


def test_submit_task_401_without_token():
    client = TestClient(status_api.app)

    response = client.post("/tasks", json={"prompt": "hello"})

    assert response.status_code == 401


def test_submit_task_400_missing_prompt():
    with TestClient(status_api.app) as client:
        response = client.post("/tasks", json={}, headers=AUTH)

    assert response.status_code == 400


def test_task_status_404_for_unknown_job():
    with TestClient(status_api.app) as client:
        response = client.get("/tasks/no-such-job/status", headers=AUTH)

    assert response.status_code == 404


def test_submit_task_poll_status_and_report_round_trip(monkeypatch):
    """#1654: exercises the real Starlette app end to end -- POST /tasks,
    poll GET /tasks/{id}/status to completion, then GET .../report -- with
    only the orchestrator boundary (src.sdk.run) faked, not the SDK's own
    job-tracking logic."""

    async def fake_run(prompt: str) -> Dict[str, Any]:
        return {"report": f"findings for {prompt}"}

    monkeypatch.setattr(sdk, "run", fake_run)

    with TestClient(status_api.app) as client:
        submit_response = client.post(
            "/tasks", json={"prompt": "durable topic"}, headers=AUTH
        )
        assert submit_response.status_code == 202
        job_id = submit_response.json()["job_id"]

        status_json = None
        for _ in range(50):
            status_response = client.get(f"/tasks/{job_id}/status", headers=AUTH)
            assert status_response.status_code == 200
            status_json = status_response.json()
            if status_json["status"] == "completed":
                break
            time.sleep(0.01)
        assert status_json["status"] == "completed"

        report_response = client.get(f"/tasks/{job_id}/report", headers=AUTH)
        assert report_response.status_code == 200
        assert report_response.json() == {"report": "findings for durable topic"}


def test_task_report_409_while_running(monkeypatch):
    import asyncio

    stuck = asyncio.Event()

    async def hanging_run(prompt: str) -> Dict[str, Any]:
        await stuck.wait()
        return {}

    monkeypatch.setattr(sdk, "run", hanging_run)

    with TestClient(status_api.app) as client:
        submit_response = client.post(
            "/tasks", json={"prompt": "slow topic"}, headers=AUTH
        )
        job_id = submit_response.json()["job_id"]

        report_response = client.get(f"/tasks/{job_id}/report", headers=AUTH)

        assert report_response.status_code == 409
