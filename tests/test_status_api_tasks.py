"""Anvil Phase B-2: POST /tasks submit -> GET status -> GET report, end to end."""

import asyncio
import time

import pytest
from starlette.testclient import TestClient

from src.web import status_api

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("STATUS_API_TOKEN", "test-token")
    status_api._tasks.clear()


def test_submit_task_requires_auth():
    client = TestClient(status_api.app)
    response = client.post("/tasks", json={"prompt": "hi"})
    assert response.status_code == 401


def test_submit_task_requires_prompt():
    client = TestClient(status_api.app)
    response = client.post("/tasks", json={}, headers=AUTH)
    assert response.status_code == 400


def test_task_status_404_for_unknown_job():
    client = TestClient(status_api.app)
    response = client.get("/tasks/no-such-id/status", headers=AUTH)
    assert response.status_code == 404


def test_submit_status_report_end_to_end(monkeypatch):
    async def fake_run(prompt):
        return {"success": True, "content": f"answer to: {prompt}"}

    monkeypatch.setattr("src.sdk.run", fake_run)

    with TestClient(status_api.app) as client:
        submit = client.post("/tasks", json={"prompt": "what is 2+2"}, headers=AUTH)
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]
        assert submit.json()["status"] == "running"

        deadline = time.time() + 5
        status = None
        while time.time() < deadline:
            status = client.get(f"/tasks/{job_id}/status", headers=AUTH)
            if status.json()["status"] != "running":
                break
            time.sleep(0.05)
        assert status is not None
        assert status.json()["status"] == "completed"

        report = client.get(f"/tasks/{job_id}/report", headers=AUTH)
        assert report.status_code == 200
        assert report.json() == {"success": True, "content": "answer to: what is 2+2"}


def test_report_409_while_still_running():
    status_api._tasks["job-running"] = {
        "status": "running",
        "prompt": "x",
        "submitted_at": "now",
    }
    client = TestClient(status_api.app)
    response = client.get("/tasks/job-running/report", headers=AUTH)
    assert response.status_code == 409


def test_report_500_on_task_failure():
    status_api._tasks["job-failed"] = {
        "status": "failed",
        "error": "boom",
        "prompt": "x",
        "submitted_at": "now",
    }
    client = TestClient(status_api.app)
    response = client.get("/tasks/job-failed/report", headers=AUTH)
    assert response.status_code == 500


if __name__ == "__main__":
    print("run via pytest")
